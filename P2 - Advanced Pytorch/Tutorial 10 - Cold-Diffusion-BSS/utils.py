import numpy as np
import torch
import torchaudio
from torch import nn
from tqdm import tqdm

class ColdDiffusion:
    def __init__(
        self,
        beta_start = 1e-4,
        beta_end = 0.2,
        noise_steps = 20,
        prediction_type = "noise"
    ):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.noise_steps = noise_steps
        self.beta = self.get_noise_schedule()
        self.alpha = 1 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)

        self.prediction_type = prediction_type

    def get_noise_schedule(self):
        """
            RETURNS
            -------
            beta: torch.Tensor
                The noise schedule - Shape: (noise_steps, )
        """
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)  # Noise schedule

    def get_noised_inputs(
        self, 
        vocals: torch.Tensor, 
        mixture: torch.Tensor, 
        t: int
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
        """
        alpha_cumprods = self.alpha_cumprod[t].to(vocals.device)          # Shape: (batch_size, )
        noised_inputs = torch.sqrt(alpha_cumprods[:, None, None]) * vocals + torch.sqrt(1.0 - alpha_cumprods[:, None, None]) * mixture
        return noised_inputs
    
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
    
    def separate(
        self,
        model: nn.Module,
        mixture: torch.Tensor
    ):
        """
            ARGUMENTS
            ---------
            model: nn.Module
                The model to use for separation of vocals from mixture

            mixture: torch.Tensor
                The mixture tensor - Shape: (batch_size, channels, audio_length)
            
            RETURNS
            -------
            vocals_pred: torch.Tensor
                The predicted vocals generated - Shape: (batch_size, channels, audio_length)
        """
        model.eval()
        with torch.no_grad():
            vocals_pred = mixture.clone()
            for t in tqdm(reversed(range(0, self.noise_steps)), position=0):                        # Starts from noise_steps-1 to 0
                timesteps = (t * torch.ones(vocals_pred.shape[0])).long().to(vocals_pred.device)    # Long is necessary for indexing from nn.Embedding
                alpha_t = self.alpha[timesteps.cpu()][:, None, None].to(vocals_pred.device)                 # Shape: (batch_size, 1, 1)
                alpha_cumprod_t = self.alpha_cumprod[timesteps.cpu()][:, None, None].to(vocals_pred.device) # Shape: (batch_size, 1, 1)
                
                if self.prediction_type=="noise":
                    pred_noise = model(vocals_pred, timesteps)  # Shape: (batch_size, channels, audio_length)
                    
                    # vocals_pred: Shape - (batch_size, channels, audio_length)
                    vocals_pred = (vocals_pred - ((1 - alpha_t)/torch.sqrt(1 - alpha_cumprod_t)) * pred_noise) * (1/torch.sqrt(alpha_t))

                    # Deviation parameter is omitted since it is a deterministic model, not probabilistic like in DDPM - it ends up performing worse
                    # If the model is probabilistic, then adding the deviation parameter helps.
                elif self.prediction_type=="mean":
                    vocals_pred = model(vocals_pred, timesteps)
                else:
                    raise ValueError("Invalid prediction type. Choose from 'noise' or 'mean'")

        model.train()
        vocals_pred.clamp_(-1, 1)           # Limiting the values between -1 and 1
        return vocals_pred

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