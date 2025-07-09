import torch
import numpy as np
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim

from diffwave import DiffWave
from augmentation_utils import pitch_augment, time_augment, phase_augment
from utils import ColdDiffusion, Metrics
from dataloader import MUSDBHQ18
from tqdm import tqdm
import wandb

from mir_eval.separation import bss_eval_sources

# Hyperparameters
learning_rate = 1e-4
warm_up_steps = 22500
gamma = 0.999995
batch_size = 2
num_epochs = 600

# Model hyperparameters
in_channels = 2
time_embedding_dim = 128
mid_channels = 64
residual_input_dim = 512
n_residual_layers = 30
val_step_check = 22500

beta_start = 1e-4
beta_end = 0.2
noise_steps = 20

# Logging into wandb
wandb.login()

# Initialize Weights and Biases
run = wandb.init(project="cold-diffusion-bss",
           config={
               "learning_rate": learning_rate,
               "warm_up_steps": warm_up_steps,
               "gamma": gamma,
               "batch_size": batch_size,
               "num_epochs": num_epochs,
               "in_channels": in_channels,
               "time_embedding_dim": time_embedding_dim,
               "mid_channels": mid_channels,
               "residual_input_dim": residual_input_dim,
               "n_residual_layers": n_residual_layers,
               "beta_start": beta_start,
               "beta_end": beta_end,
               "noise_steps": noise_steps
           })

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Device: ", device)

# Load dataset and dataloader
train_data = torch.load("data/train.pt")
test_data = torch.load("data/test.pt")
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = DiffWave(
    time_embedding_dim=time_embedding_dim,
    in_channels=in_channels,
    mid_channels=mid_channels,
    residual_input_dim=residual_input_dim,
    n_residual_layers=n_residual_layers,
    device=device
).to(device)

# Initialize diffusion
diffusion = ColdDiffusion(
    model=model,
    beta_start=beta_start,
    beta_end=beta_end,
    noise_steps=noise_steps
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(diffusion.parameters(), lr=learning_rate)
scheduler1 = optim.lr_scheduler.LinearLR(optimizer, 1e-7, 1, warm_up_steps)
scheduler2 = optim.lr_scheduler.ExponentialLR(optimizer, gamma)
scheduler = optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], [warm_up_steps])

curr_step = 0

metrics = Metrics()

# Training loop
for epoch in range(num_epochs):
    diffusion.train()
    print(f"Epoch {epoch+1}:")
    pbar = tqdm(train_dataloader)
    for step, (vocals, mixture) in enumerate(pbar):
        batch_size = vocals.shape[0]
        new_mixture = mixture.clone()
        new_vocals = vocals.clone()
        for i in range(2):
            if i==1:
                new_mixture, new_vocals = pitch_augment(mixture, vocals, sample_rate=16000)
            # elif i==2:
            #     original_time = vocals.shape[-1]
            #     vocals, mixture, _ = time_augment(vocals.cpu(), mixture.cpu(), mixture.cpu())
            #     rand_start = torch.randint(0, vocals[0].shape[-1] - original_time, (1,))[0].item()
            #     vocals = torch.stack([vocal[:, rand_start:rand_start+original_time] for vocal in vocals])
                # mixture = torch.stack([mix[:, rand_start:rand_start+original_time] for mix in mixture])

            new_vocals = new_vocals.to(device)
            new_mixture = new_mixture.to(device)
            
            time_steps = diffusion.sample_timesteps(batch_size)
            
            optimizer.zero_grad()

            x_t, accomp = diffusion.get_noised_inputs(new_vocals, new_mixture, time_steps)
            # accomp = new_mixture - new_vocals

            pred_noise = diffusion.pred_noise(x_t, time_steps)
            loss = criterion(pred_noise, accomp)

            
            loss.backward()
            optimizer.step()
            scheduler.step()

            curr_step+=1

            pbar.set_postfix(MSE = loss.item())
            

            if curr_step%val_step_check==0:
                torch.save(diffusion.state_dict(), f"models/diffusion_{curr_step}.pth")
                torch.save(optimizer.state_dict(), f"models/diffusion_optimizer_{curr_step}.pth")
                torch.save(scheduler.state_dict(), f"models/diffusion_scheduler_{curr_step}.pth")
                # Validation loop
                diffusion.eval()
                avg_val_values = np.array([0.0, 0.0, 0.0])
                with torch.no_grad():
                    pbar_eval = tqdm(test_dataloader)
                    total_len = 0
                    for val_step, (vocals, mixture) in enumerate(pbar_eval):
                        vocals = vocals.to(device)
                        mixture = mixture.to(device)

                        batch_size = vocals.shape[0]

                        vocals_pred = diffusion(mixture)
                        vocals_pred = vocals_pred * np.max(np.abs(vocals.cpu().numpy()))/np.max(np.abs(vocals_pred.cpu().numpy()))

                        # sdr, sir, sar, _ = bss_eval_sources(vocals.cpu().numpy(), vocals_pred.cpu().numpy())
                        sdr = metrics.sdr(vocals.cpu().numpy(), vocals_pred.cpu().numpy())
                        avg_val_values[0] += np.sum(sdr)
                        total_len += batch_size
                        # avg_val_values[1] += sir
                        # avg_val_values[2] += sar
                        pbar_eval.set_postfix(SDR = np.mean(sdr))
                    
                    avg_val_values/=total_len
                diffusion.train()

                run.log({
                    "epoch": epoch+1,
                    "LR": scheduler.get_last_lr()[0],
                    "step": step+1,
                    "curr_step": curr_step,
                    "MSE loss": loss.item(),
                    "Validation SDR": avg_val_values[0],
                    # "Validation SIR": avg_val_values[1].item(),
                    # "Validation SAR": avg_val_values[2].item(),
                })
            else:
                run.log({
                    "epoch": epoch+1,
                    "LR": scheduler.get_last_lr()[0],
                    "step": step+1,
                    "curr_step": curr_step,
                    "MSE loss": loss.item()
                })
                torch.save(diffusion.state_dict(), f"models/ckpt.pth")
                torch.save(optimizer.state_dict(), f"models/ckpt_optimizer.pth")

        
        