import torch
import numpy as np
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim

from diffwave import DiffWave
from utils import ColdDiffusion, Metrics
from dataloader import MUSDBHQ18
from tqdm import tqdm
import wandb

from mir_eval.separation import bss_eval_sources

# Hyperparameters
learning_rate = 1e-4
batch_size = 2
num_epochs = 500

# Model hyperparameters
in_channels = 2
time_embedding_dim = 128
mid_channels = 64
residual_input_dim = 512
n_residual_layers = 30
val_step_check = 5000

beta_start = 1e-4
beta_end = 0.2
noise_steps = 20

prediction_type = "noise"

# Logging into wandb
wandb.login()

# Initialize Weights and Biases
run = wandb.init(project="cold-diffusion-bss",
           config={
               "learning_rate": learning_rate,
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

# Initialize diffusion
diffusion = ColdDiffusion(
    beta_start=beta_start,
    beta_end=beta_end,
    noise_steps=noise_steps,
    prediction_type = prediction_type
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

# model.load_state_dict(torch.load("./models/ckpt_old.pth"))

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

curr_step = 0

metrics = Metrics()

# Training loop
for epoch in range(num_epochs):
    model.train()
    print(f"Epoch {epoch+1}:")
    pbar = tqdm(train_dataloader)
    for step, (vocals, mixture) in enumerate(pbar):
        
        vocals = vocals.to(device)
        mixture = mixture.to(device)

        batch_size = vocals.shape[0]

        time_steps = diffusion.sample_timesteps(batch_size)
        
        optimizer.zero_grad()
        if prediction_type=="noise":
            x_t = diffusion.get_noised_inputs(vocals, mixture, time_steps)
            
            noise = x_t - vocals        # Accompaniment
            pred_noise = model(vocals, time_steps)
            loss = criterion(pred_noise, noise)
        elif prediction_type=="mean":
            x_t = diffusion.get_noised_inputs(vocals, mixture, time_steps)
            x_t_minus_1 = diffusion.get_noised_inputs(vocals, mixture, time_steps - 1)
            
            pred_x_t_minus_1 = model(vocals, time_steps)
            loss = criterion(pred_x_t_minus_1, x_t_minus_1)
        else:
            raise ValueError("Invalid prediction type. Choose from 'noise' or 'mean'")

        
        loss.backward()
        optimizer.step()

        curr_step+=1

        pbar.set_postfix(MSE = loss.item())
        

        if curr_step%val_step_check==0:
            torch.save(model.state_dict(), f"models/model_{curr_step}.pth")
            torch.save(optimizer.state_dict(), f"models/model_optimizer_{curr_step}.pth")
            # Validation loop
            model.eval()
            avg_val_values = np.array([0.0, 0.0, 0.0])
            with torch.no_grad():
                for val_step, (vocals, mixture) in enumerate(test_dataloader):
                    vocals = vocals.to(device)
                    mixture = mixture.to(device)

                    batch_size = vocals.shape[0]

                    vocals_pred = diffusion.separate(model, mixture)

                    # sdr, sir, sar, _ = bss_eval_sources(vocals.cpu().numpy(), vocals_pred.cpu().numpy())
                    sdr = metrics.sdr(vocals.cpu().numpy(), vocals_pred.cpu().numpy())
                    avg_val_values[0] += np.mean(sdr)
                    # avg_val_values[1] += sir
                    # avg_val_values[2] += sar
                
                avg_val_values/=len(test_dataloader)
            model.train()

            run.log({
                "epoch": epoch+1,
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
                "step": step+1,
                "curr_step": curr_step,
                "MSE loss": loss.item()
            })
            torch.save(model.state_dict(), f"models/ckpt.pth")
            torch.save(optimizer.state_dict(), f"models/ckpt_optimizer.pth")

        
        