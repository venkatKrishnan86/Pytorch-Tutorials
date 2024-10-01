import os
import torch
from torch import nn
from matplotlib import pyplot as plt
from tqdm import tqdm
import logging

# from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class DiffusionToolbox:
    """
        Tools we need to perform diffusion
    """
    def __init__(
        self,
        noise_steps = 1000,
        beta_start = 1e-4,
        beta_end = 0.02,
        img_size = 64,
        device = "cpu",
        logger = True
    ) -> None:
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.logger = logger

        # Refer to the paper
        self.beta = self.prepare_noise_schedule()
        self.alpha = 1 - self.beta
        self.alpha_dash = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        # Returning a linear schedule
        # We can also use cosine schedule by OpenAI
        return torch.linspace(
            start=self.beta_start, 
            end=self.beta_end, 
            steps=self.noise_steps
        )
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n, ))
    
    def get_noised_images(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor
    ):
        """
            Arguments
            ---------
            x: The input x_0 (Shape: [batch_size, channels, img_size, img_size])
            t: List of time indices required as a tensor
        """
        # x_t = \sqrt{alphadash_t}x_0 + \sqrt{1 - alphadash_t} \eps
        sqrt_alpha_dash = torch.sqrt(self.alpha_dash[t])
        sqrt_alpha_dash = sqrt_alpha_dash[:, None, None, None].to(self.device) # Broadcasting to 4 dimensions
        sqrt_one_minus_alpha_dash = torch.sqrt(1. - self.alpha_dash[t])
        sqrt_one_minus_alpha_dash = sqrt_one_minus_alpha_dash[:, None, None, None].to(self.device)
        eps = torch.randn_like(x).to(self.device)
        return sqrt_alpha_dash*x + sqrt_one_minus_alpha_dash * eps, eps
    
    def sample(
        self, 
        eps_model: nn.Module, 
        n: int
    ):
        if self.logger:
            logging.info(f"Sampling {n} new images...")
        eps_model.eval()
        with torch.no_grad():
            # Getting n 3-channel gaussian noise image samples
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)

            # position in tqdm is used to mention which progress bar it is
            # useful to offset different tqdm bars to different lines
            for i in tqdm(reversed(range(0, self.noise_steps)), position=0):
                t = (i * torch.ones(n)).long().to(self.device)
                alpha = self.alpha[t.cpu()][:, None, None, None].to(self.device)
                beta = self.beta[t.cpu()][:, None, None, None].to(self.device)
                alpha_dash = self.alpha_dash[t.cpu()][:, None, None, None].to(self.device)

                eps_pred = eps_model(x, t)

                if i>0:
                    z = torch.randn_like(x).to(self.device)
                else:
                    z = torch.zeros_like(x).to(self.device)
                
                # Reverse Diffusion to obtain final x at t=0
                x = (1.0/torch.sqrt(alpha)) * (x - ((1 - alpha)/(torch.sqrt(1 - alpha_dash))) * eps_pred) + torch.sqrt(beta) * z

        eps_model.train()
        x = (x.clamp(-1, 1) + 1)/2.0      # Limiting the values between -1 and 1
        x = (x*255).type(torch.uint8)   # Converting to pixel values
        return x

if __name__=="__main__":
    diffusion = DiffusionToolbox()
    t = torch.randint(0, 1000, (2, ))
    print(t)
    print(diffusion.get_noised_images(torch.randn((2, 1, 2, 2)), t))


