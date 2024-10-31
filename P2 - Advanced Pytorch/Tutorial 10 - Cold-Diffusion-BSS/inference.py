import torch
import numpy as np
from torch.utils.data import DataLoader
from diffwave import DiffWave
from utils import ColdDiffusion
from tqdm import tqdm

from dataloader import MUSDBHQ18

from mir_eval.separation import bss_eval_sources

device = "cpu"
batch_size = 2
test_data = torch.load("data/test.pt")
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

learning_rate = 1e-4
num_epochs = 1000

# Model hyperparameters
in_channels = 2
time_embedding_dim = 128
mid_channels = 64
residual_input_dim = 512
n_residual_layers = 30

beta_start = 1e-4
beta_end = 0.2
noise_steps = 20

# Initialize diffusion
diffusion = ColdDiffusion(
    beta_start=beta_start,
    beta_end=beta_end,
    noise_steps=noise_steps
)

# Initialize model, loss function, and optimizer
model = DiffWave(
    time_embedding_dim=time_embedding_dim,
    in_channels=in_channels,
    mid_channels=mid_channels,
    residual_input_dim=residual_input_dim,
    n_residual_layers=n_residual_layers,
    device=device
).to(device)

model.load_state_dict(torch.load("models/ckpt.pth"))

model.eval()
avg_val_values = [0.0, 0.0, 0.0]

def sdr(references, estimates):
    # compute SDR for one song
    delta = 1e-7  # avoid numerical errors
    num = np.sum(np.square(references), axis=(1, 2))
    den = np.sum(np.square(references - estimates), axis=(1, 2))
    num += delta
    den += delta
    return 10 * np.log10(num / den)

with torch.no_grad():
    pbar = enumerate(test_dataloader)
    
    for val_step, (vocals, mixture) in pbar:
        vocals = vocals.to(device)
        mixture = mixture.to(device)

        batch_size = vocals.shape[0]

        vocals_pred = diffusion.separate(model, mixture)

        print(vocals.shape, vocals_pred.shape)

        sdr_val = sdr(vocals.cpu().numpy(), vocals_pred.cpu().numpy())
        # avg_val_values[0] += sdr_val
        # avg_val_values[1] += sir
        # avg_val_values[2] += sar
        print(f"Validation Step: {val_step+1}/{len(test_dataloader)}")
        print(f"Validation SDR: {sdr_val}")
        # print(f"Validation SIR: {avg_val_values[1]}")
        # print(f"Validation SAR: {avg_val_values[2]}")
        print()
    
    avg_val_values/=len(test_dataloader)
    print(f"Validation SDR: {avg_val_values[0]}")
    print(f"Validation SIR: {avg_val_values[1]}")
    print(f"Validation SAR: {avg_val_values[2]}")