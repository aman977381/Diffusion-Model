import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import *
from module import UNet
from torch import optim
from tqdm import tqdm
import logging
from model import SimpleUnet  # Import the SimpleUnet model
from torchvision.utils import make_grid
from io import BytesIO
from torch.utils.tensorboard import SummaryWriter
logging.basicConfig(format = "%(asctime)s - %(levelname)s :%(message)s", level = logging.INFO, datefmt="%I:%M:%S")

writer_graph = SummaryWriter(f"logs/graph")

class Diffusion:
    def __init__ (self, noise_steps = 1000, beta_start = 1e-4, beta_end = 0.02, img_size = 64, device = "cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1-self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim = 0)

    
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    
    def sample_timesteps(self,n):
        return torch.randint(low = 1, high = self.noise_steps, size = (n,))
    
    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position = 0):

                t = (torch.ones(n)*i).long().to(self.device)

                predicted_noise = model(x,t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(x)

                else:
                    noise = torch.zeros_like(x)

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        
        model.train()
        x = (x.clamp(-1,1)+1)/2
        x = (x*255).type(torch.uint8)
        return x
    


def train(args):
    setup_logging(args.run_name)
    device = args.device
    train_loader, val_loader = get_data(args)
    model = SimpleUnet().to(device)  # Use SimpleUnet
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(train_loader)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')  # Track best validation loss

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        
        # Training phase
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        epoch_loss = 0
        
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(MSE=loss.item())
            writer_graph.add_scalar("Training MSE", loss.item(), global_step=epoch * l + i)


        # Average training loss for the epoch
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        
        with torch.no_grad():
            for images, _ in val_pbar:
                images = images.to(device)
                t = diffusion.sample_timesteps(images.shape[0]).to(device)
                x_t, noise = diffusion.noise_images(images, t)
                predicted_noise = model(x_t, t)
                val_loss = mse(noise, predicted_noise)
                epoch_val_loss += val_loss.item()
                val_pbar.set_postfix(MSE=val_loss.item())

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        writer_graph.add_scalars("Validation loss vs Training loss ", {"Average Validation MSE": avg_val_loss, "Average Training MSE": avg_train_loss}, global_step=epoch)

        # Print epoch results
        logging.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join("models", args.run_name, "best_model.pt"))
            logging.info(f"New best model saved with validation loss: {best_val_loss:.4f}")

        # Save results and latest model
        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, os.path.join("results", args.run_name, f"epoch_{epoch+1}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, "latest_model.pt"))

    # Plot and save loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join("results", args.run_name, 'loss_curves.png'))
    plt.close()




def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "Diffusion_Model"
    args.epochs = 200
    args.batch_size = 64
    args.image_size = 64
    args.dataset_path = r"./artifacts"
    args.device = "cuda"
    args.lr = 3e-5
    train(args)

if __name__ == '__main__':
    launch()