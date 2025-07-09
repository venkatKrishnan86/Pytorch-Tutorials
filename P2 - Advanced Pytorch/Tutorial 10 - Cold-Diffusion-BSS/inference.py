import numpy as np
import torch
import torchaudio
from torch import nn
from tqdm import tqdm
from diffwave import DiffWave

class ColdDiffusion(nn.Module):
    def __init__(
            self,
            model: DiffWave,
            beta_start = 1e-4,
            beta_end = 0.2,
            noise_steps = 20
        ):
        super(ColdDiffusion, self).__init__()
        self.model = model
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.noise_steps = noise_steps
        self.beta = self._get_noise_schedule()
        self.alpha = 1 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)

    def _get_noise_schedule(self):
        """
            RETURNS
            -------
            beta: torch.Tensor
                The noise schedule - Shape: (noise_steps, )
        """
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)  # Noise schedule

    def pred_noise(self, signal, timesteps):
        """
            Arguments
            ---------
            signal: torch.Tensor
                The input tensor - Shape: (batch_size, in_channels, audio_length)

            timesteps: torch.Tensor
                The time step indices - Shape: (batch_size, )

            Returns
            -------
            pred_noise: torch.Tensor
                The output tensor - Shape: (batch_size, in_channels, audio_length)
        """
        return self.model(signal, timesteps)

    def pred_mean(self, signal, noise, timestep):
        """
            Computes the mean of the denoised signal

            ARGUMENTS
            ---------
            signal: torch.Tensor
                The input signal tensor - Shape: (batch_size, in_channels, audio_length)

            noise: torch.Tensor
                The estimated noise tensor - Shape: (batch_size, in_channels, audio_length)

            timestep: int
                The time step index to compute the mean

            RETURNS
            -------
            mean: torch.Tensor
                The estimated mean of denoised signal tensor - Shape: (batch_size, in_channels, audio_length)
        """
        alpha = self.alpha[timestep]
        alpha_bar = self.alpha_cumprod[timestep]
        mean = (1.0/ torch.sqrt(alpha)) * (signal - (1 - alpha) / torch.sqrt(1 - alpha_bar) * noise)
        return mean

    def forward(self, signal):
        """
            Generating denoised signal from the input signal

            ARGUMENTS
            ---------
            signal: torch.Tensor
                The input signal tensor - Shape: (batch_size, in_channels, audio_length)

            RETURNS
            -------
            separated_signal: torch.Tensor
                The separated signal tensor - Shape: (batch_size, in_channels, audio_length)
        """
        base = torch.ones(signal.shape[0]).to(signal.device)
        for t in reversed(range(self.noise_steps)):
            noise = self.pred_noise(signal, t * base)
            signal = self.pred_mean(signal, noise, torch.tensor(t, dtype = torch.long))
        return signal

    def get_noised_inputs(
        self, 
        vocals: torch.Tensor, 
        mixture: torch.Tensor, 
        t: torch.Tensor
    ):
        """
            ARGUMENTS
            ---------
            vocals: torch.Tensor
                The vocals tensor (x_0) - Shape: (batch_size, channels, audio_length)

            mixture: torch.Tensor
                The mixture tensor (x_T) - Shape: (batch_size, channels, audio_length)

            t: torch.Tensor
                Time step indices - Shape: (batch_size, )

            RETURNS
            -------
            noised_inputs: torch.Tensor
                The noised inputs (x_t) - Shape: (batch_size, channels, audio_length)

            noise: torch.Tensor
                The noise tensor (x_t - x_0) - Shape: (batch_size, channels, audio_length)
        """
        alpha_cumprods = self.alpha_cumprod[t].to(vocals.device)          # Shape: (batch_size, )
        noised_inputs = torch.sqrt(alpha_cumprods[:, None, None]) * vocals + torch.sqrt(1.0 - alpha_cumprods[:, None, None]) * mixture
        noise = mixture - vocals
        return noised_inputs, noise
    
    def sample_timesteps(self, n):
        """
            ARGUMENTS
            ---------
            n: int
                Number of samples to generate
            
            RETURNS
            -------
            timesteps: torch.Tensor
                The time steps to sample from a uniform distribution - Shape: (n, )
        """
        return torch.randint(low=1, high=self.noise_steps, size=(n, ))

class Metrics:
    def __init__(self):
        self.delta = 1e-7  # avoid numerical errors
    
    def sdr(self, references, estimates):
        # compute SDR for one song
        num = np.sum(np.square(references), axis=(1, 2))
        den = np.sum(np.square(references - estimates), axis=(1, 2))
        num += self.delta
        den += self.delta
        return 10 * np.log10(num / den)

if __name__=="__main__":
    cd = ColdDiffusion(noise_steps=5)
    print(cd.beta)
    print(cd.alpha)
    print(cd.alpha_cumprod)
    print(cd.alpha_cumprod[-1])
    print(cd.alpha_cumprod[0])

    print("Done!")