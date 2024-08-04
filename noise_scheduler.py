import torch

class NoiseScheduler:
    def __init__(self, timesteps, beta_start, beta_end):
        self.timesteps = timesteps
        self.beta = torch.linspace(beta_start, beta_end, timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def forward_diffusion(self, x_0, t):
        """
        Forward diffusion process (adding noise).

        Args:
            x_0: Original data tensor.
            t: Time step (or an array of time steps).

        Returns:
            x_t: Noised data at time t.
            noise: The Gaussian noise added.
        """
        # Ensure t is a tensor and expand dimensions
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        if len(t.shape) == 0:
            t = t.unsqueeze(0)

        # Add batch dimension to t and alphas
        alphas_t = self.alpha_bar[t].unsqueeze(1)
        alphas_t_sqrt = alphas_t.sqrt()
        one_minus_alphas_t_sqrt = (1 - alphas_t).sqrt()

        # Sample Gaussian noise
        noise = torch.randn_like(x_0)

        # Compute x_t
        x_t = alphas_t_sqrt * x_0 + one_minus_alphas_t_sqrt * noise

        return x_t, noise