import torch
import torch.nn.functional as F
from tqdm import tqdm
import math

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class Diffusion:
    def __init__(self, timesteps=1000, data_mean=0.0, data_std=1.0, device='cpu'):
        self.timesteps = timesteps
        self.device = device
        self.data_mean = data_mean
        self.data_std = data_std

        self.betas = cosine_beta_schedule(timesteps).to(self.device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def _get_view_shape(self, x):
        if x.dim() == 2:
            return (-1, 1)
        elif x.dim() == 3:
            return (-1, 1, 1)
        elif x.dim() == 4:
            return (-1, 1, 1, 1)
        else:
            raise ValueError(f"Unsupported tensor dimension: {x.dim()}")

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        view_shape = self._get_view_shape(x_0)

        return (
            self.sqrt_alphas_cumprod[t].view(*view_shape) * x_0 +
            self.sqrt_one_minus_alphas_cumprod[t].view(*view_shape) * noise
        )

    def q_sample_coarse_target(self, x_0, t, x_c, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        view_shape = self._get_view_shape(x_0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(*view_shape)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(*view_shape)

        mean = sqrt_alphas_cumprod_t * x_0 + (1 - sqrt_alphas_cumprod_t) * x_c

        return mean + sqrt_one_minus_alphas_cumprod_t * noise

    def _get_mean_variance(self, x_t, t, pred_x0):
        view_shape = self._get_view_shape(x_t)
        coef1 = self.posterior_mean_coef1[t].view(*view_shape)
        coef2 = self.posterior_mean_coef2[t].view(*view_shape)
        mean = coef1 * pred_x0 + coef2 * x_t
        variance = self.posterior_variance[t].view(*view_shape)
        return mean, variance

    @torch.no_grad()
    def ddim_sample_loop(self, model, shape, *args, **kwargs):
        num_steps = kwargs.pop('num_steps', 50)
        eta = kwargs.pop('eta', 0.0)
        x_c = kwargs.pop('x_c', None)

        guidance_scale = kwargs.pop('guidance_scale', 1.0)
        show_progress = kwargs.pop('show_progress', True)

        query_batch_size = kwargs.pop('query_batch_size', None)

        if x_c is None:
            img = torch.randn(shape, device=self.device)
            desc = "DDIM Sampling (From Noise)"
        else:
            start_time = self.timesteps - 1
            view_shape_start = self._get_view_shape(x_c)
            sqrt_1m_at_start = self.sqrt_one_minus_alphas_cumprod[start_time].view(*view_shape_start)
            img = x_c + sqrt_1m_at_start * torch.randn(shape, device=self.device)
            desc = f"DDIM Sampling (From Coarse, CFG={guidance_scale})"

        t_uniform_norm = torch.linspace(1.0, 0.0, steps=num_steps + 1, device=self.device)
        t_distorted_norm = 2 * t_uniform_norm - t_uniform_norm.pow(2)
        times = (t_distorted_norm * self.timesteps) - 1
        times = list(times.int().tolist())
        times[0] = self.timesteps - 1
        times[-1] = -1
        time_pairs = list(zip(times[:-1], times[1:]))

        x_self = None
        for time, time_prev in tqdm(time_pairs, desc=desc, disable=not show_progress):
            t = torch.full((shape[0],), time, device=self.device, dtype=torch.long)

            current_kwargs = kwargs.copy()
            if x_c is not None and 'x_c' not in current_kwargs:
                 current_kwargs['x_c'] = x_c
            current_kwargs['x_self'] = x_self

            def run_model_batch_aware(input_img, input_t, model_kwargs):
                if query_batch_size is None:
                    out, _ = model(input_img, input_t, *args, **model_kwargs)
                    return out
                else:
                    num_total_nodes = shape[0]
                    num_batches = (num_total_nodes + query_batch_size - 1) // query_batch_size
                    batch_outputs = []

                    for i in range(num_batches):
                        batch_start = i * query_batch_size
                        batch_end = min((i + 1) * query_batch_size, num_total_nodes)

                        img_batch = input_img[batch_start:batch_end]
                        t_batch = input_t[batch_start:batch_end]

                        batch_kwargs = {}
                        for k, v in model_kwargs.items():
                            if torch.is_tensor(v) and v.shape[0] == num_total_nodes:
                                batch_kwargs[k] = v[batch_start:batch_end]
                            else:
                                batch_kwargs[k] = v

                        out_batch, _ = model(img_batch, t_batch, *args, **batch_kwargs)
                        batch_outputs.append(out_batch)

                    return torch.cat(batch_outputs, dim=0)

            if guidance_scale == 1.0:
                pred_x0 = run_model_batch_aware(img, t, current_kwargs)
            else:

                pred_x0_cond = run_model_batch_aware(img, t, current_kwargs)

                uncond_kwargs = current_kwargs.copy()
                if 'history_embeddings' in uncond_kwargs:
                    uncond_kwargs['history_embeddings'] = torch.zeros_like(uncond_kwargs['history_embeddings'])
                if 'x_c' in uncond_kwargs:
                    uncond_kwargs['x_c'] = torch.zeros_like(uncond_kwargs['x_c'])

                pred_x0_uncond = run_model_batch_aware(img, t, uncond_kwargs)

                pred_x0 = pred_x0_uncond + guidance_scale * (pred_x0_cond - pred_x0_uncond)

            x_self = pred_x0
            if time_prev < 0:
                img = pred_x0
                continue

            alpha_prod_t = self.alphas_cumprod[time]
            alpha_prod_t_prev = self.alphas_cumprod[time_prev]

            view_shape = self._get_view_shape(img)
            sqrt_at = torch.sqrt(alpha_prod_t).view(*view_shape)
            sqrt_at_prev = torch.sqrt(alpha_prod_t_prev).view(*view_shape)
            sqrt_1m_at = torch.sqrt(1. - alpha_prod_t).view(*view_shape)
            sqrt_1m_at_prev = torch.sqrt(1. - alpha_prod_t_prev).view(*view_shape)

            if x_c is not None:
                center_t = (1. - sqrt_at) * x_c
                center_t_prev = (1. - sqrt_at_prev) * x_c
            else:
                center_t = torch.zeros_like(img)
                center_t_prev = torch.zeros_like(img)

            pred_epsilon = (img - sqrt_at * pred_x0 - center_t) / sqrt_1m_at
            img = sqrt_at_prev * pred_x0 + center_t_prev + sqrt_1m_at_prev * pred_epsilon

            if eta > 0:
                sigma = eta * (sqrt_1m_at_prev / sqrt_1m_at) * torch.sqrt(
                    torch.clamp(1. - (sqrt_at / sqrt_at_prev) ** 2, min=0.0))
                img = img + sigma * torch.randn_like(img)

        return img
