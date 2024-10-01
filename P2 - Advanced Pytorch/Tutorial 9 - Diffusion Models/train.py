import os
import torch
from torch import nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from modules import UNet
from ddpm import DiffusionToolbox
import logging
import wandb

from utils import *

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader: DataLoader = get_data(args)

    model = UNet(device=device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)
    criterion = nn.MSELoss()
    diffusion = DiffusionToolbox(img_size=args.image_size, device=device)
    
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for step, (images, _) in enumerate(pbar):
            # logging.info(f"Step {step}:")
            images: torch.Tensor = images.to(device)
            
            # Training steps as discussed in the paper
            batch_size = images.shape[0]
            t = diffusion.sample_timesteps(batch_size)          # Shape: (batch_size, )
            x_t, eps = diffusion.get_noised_images(images, t)
            eps_pred = model(x_t, t)
            loss: torch.Tensor = criterion(eps, eps_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE = loss.item())
            args.run.log({
                "epoch": epoch,
                "step": step,
                "MSE loss": loss.item()
            })
        
        sampled_images = diffusion.sample(model, n=batch_size)
        save_images(os.path.join("results", args.run_name, f"{epoch}.jpg"), device, sampled_images)
        torch.save(model.state_dict(), os.path.join("models", args.run_name, "ckpt.pth"))


def launch():
    import argparse
    wandb.login()
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    run = wandb.init(
        project="DDPM_Uncondtional",    # set the wandb project where this run will be logged
        config={}                                  # track hyperparameters and run metadata
    )

    args.run_name = "DDPM_Uncondtional"
    args.epochs = 500
    args.batch_size = 12
    args.image_size = 64

    # Dataset: https://www.kaggle.com/datasets/arnaud58/landscape-pictures?resource=download
    args.dataset_path = "/home/venkatakrishnan/Desktop/datasets/landscape_img_dataset"
    args.use_data_path = False
    args.device = "cuda:0"
    args.run = run
    args.lr = 3e-4
    train(args)


if __name__ == '__main__':
    launch()
    # device = "cuda"
    # model = UNet().to(device)
    # ckpt = torch.load("./working/orig/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # x = diffusion.sample(model, 8)
    # print(x.shape)
    # plt.figure(figsize=(32, 32))
    # plt.imshow(torch.cat([
    #     torch.cat([i for i in x.cpu()], dim=-1),
    # ], dim=-2).permute(1, 2, 0).cpu())
    # plt.show()


