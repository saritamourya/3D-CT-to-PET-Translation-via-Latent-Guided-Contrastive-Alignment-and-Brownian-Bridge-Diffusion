
import os
import torch
import numpy as np  
from torch.cuda.amp import autocast
import torch.optim as optim
from tqdm.autonotebook import tqdm
from models.BrownianBridge.CT2PETDiffusionModel_sharedlatent import CT2PETDiffusionModel
from utils.checkpoints_utils import save_checkpoint_BB, load_autoencoder_checkpoint
from data.dataset_CACHE import CreateDataloader
from torch.utils.tensorboard import SummaryWriter
from utils import utils
from models.model_config import ModelConfig
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# EMA Class
class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.model = model
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, shadow_dict):
        self.shadow = shadow_dict

# Model Initialization

def initialize_model(model_config, device):
    model = CT2PETDiffusionModel(model_config, device)
    return model

def visualize_shift_to_shared(model, z_ct, z_shared, epoch, district, checkpoint_dir, writer=None, interval=10):
    """
    Visualizes the diffusion shift from CT latent (z_ct) to shared latent (z_shared)
    over intermediate noisy latents using t-SNE and annotates start, interval steps & end.

    Args:
        model: The diffusion model.
        z_ct (Tensor): CT latent. [B, C, D, H, W]
        z_shared (Tensor): Shared latent from AE (used as target). [B, C, D, H, W]
        epoch (int): Current epoch.
        district (str): For directory naming.
        checkpoint_dir (str): Where to save.
        writer: Optional tensorboard writer.
        interval (int): Step interval to highlight intermediate progress.
    """
    print(f"[INFO] Visualizing diffusion shift to shared latent at epoch {epoch}...")

    with torch.no_grad():
        with autocast():  # Ensure input tensors match model weights under AMP
            z_t_steps, _ = model.p_sample_loop(y=z_shared, context=z_shared, clip_denoised=False, sample_mid_step=True)

    all_latents = torch.stack(z_t_steps)  # [T, B, C, D, H, W]
    all_latents = all_latents.permute(1, 0, 2, 3, 4, 5)  # [B, T, C, D, H, W]

    for idx in range(z_ct.shape[0]):
        latents = all_latents[idx]  # [T, C, D, H, W]
        flat_latents = latents.view(latents.size(0), -1).cpu().numpy()
        z_ct_np = z_ct[idx].view(-1).cpu().numpy()
        z_shared_np = z_shared[idx].view(-1).cpu().numpy()

        trajectory = np.vstack([z_ct_np, flat_latents, z_shared_np])
        tsne = TSNE(n_components=2, perplexity=min(30, trajectory.shape[0] - 1), random_state=42)
        embedded = tsne.fit_transform(trajectory)

        plt.figure(figsize=(8, 6))
        plt.plot(embedded[:, 0], embedded[:, 1], linestyle='--', marker='o', alpha=0.2, color='gray', label='Trajectory')

        # Highlight intermediate steps
        for step in range(1, len(embedded) - 1, interval):
            plt.scatter(embedded[step, 0], embedded[step, 1], s=60, color='orange', edgecolor='black', label='Intermediate Step' if step == interval else "")

        # Start and End Points
        plt.scatter(embedded[0, 0], embedded[0, 1], color='blue', label='z_ct (start)', s=100, marker='o', edgecolor='black')
        plt.scatter(embedded[-1, 0], embedded[-1, 1], color='red', label='z_shared (target)', s=100, marker='X', edgecolor='black')
        plt.annotate("Start", (embedded[0, 0], embedded[0, 1]), textcoords="offset points", xytext=(-15, 5), ha='right', color='blue')
        plt.annotate("End", (embedded[-1, 0], embedded[-1, 1]), textcoords="offset points", xytext=(10, 5), ha='left', color='red')

        plt.title(f"Latent Shift (z_ct → shared) - Sample {idx}, Epoch {epoch}")
        plt.legend()
        plt.tight_layout()

        save_dir = os.path.join(checkpoint_dir, district, "latent_shift")
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, f"latent_shift_progressive_sample{idx}_epoch{epoch}.png")
        plt.savefig(plot_path)

        if writer:
            img = plt.imread(plot_path)
            writer.add_image(f"LatentShift_Progress/Sample_{idx}", torch.tensor(img).permute(2, 0, 1), epoch)

        plt.close()


# Training Function

def train_ct2pet_diffusion(model, autoencoder , optimizer, data_loader, device, model_config):

    num_epochs = model_config.training_params['epochs']
    model.train()
    writer = SummaryWriter(comment="eventlog_for_CT2PET_GN32_Sharedlatent_training")
    avgloss = utils.AverageLoss()
    total_counter = 0
    checkpoint_dir = model_config.checkpoint_path
    scaler = torch.cuda.amp.GradScaler()

    # EMA Configuration
    use_ema = True
    ema_decay = 0.995
    start_ema_step = 1000
    update_ema_interval = 8
    ema = EMA(model, decay=ema_decay) if use_ema else None

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_diffusion_loss = 0.0
        epoch_perceptual_loss = 0.0
        step_counter = 0
        optimizer.zero_grad()

        for batch_idx, data in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            ct_images = data['A'].to(device)
            pet_images = data['B'].to(device)
            # print(f'ct_image shape:{ct_images.shape}, pet_image shape: {pet_images.shape}')

            with torch.cuda.amp.autocast():
                mu, mu_ct, z_pet_denoised_tensor, recon_pet, diffusion_loss, log_dict = model(ct_images, pet_images)
                out_raw = model.sample(ct_images, clip_denoised=False, sample_mid_step=False).detach().cpu().squeeze(0).numpy()
                    # --- Sampling from EMA model ---
                if use_ema and total_counter >= start_ema_step:
                    ema.apply_shadow()
                    out_ema = model.sample(ct_images, clip_denoised=False, sample_mid_step=False).detach().cpu().squeeze(0).numpy()
                    ema.restore()
                else:
                    out_ema = out_raw  # fallback if EMA not yet started
                total_loss = diffusion_loss  #+recon_loss + 0.3* perceptual_loss 

                scaler.scale(total_loss).backward()
                unscaled_grads = [p.grad for p in model.parameters() if p.grad is not None]
                if len(unscaled_grads) > 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if use_ema and total_counter >= start_ema_step and total_counter % update_ema_interval == 0:
                        ema.update()  # EMA update step
                else:
                    print("[WARNING] No gradients found. Skipping optimizer step.")
                # scheduler.step(total_loss.item())

            if total_counter % 500 == 0:
                lr = optimizer.param_groups[0]['lr']
                writer.add_scalar("LearningRate", lr, total_counter)
                print(f"[INFO] Step {total_counter} - LR: {lr:.8f}")

            total_counter += 1

            epoch_loss += total_loss.item()
            epoch_diffusion_loss += diffusion_loss.item()
            step_counter += 1
            avg_total_loss = epoch_loss / step_counter
            avg_diffusion_loss = epoch_diffusion_loss / step_counter

            avgloss.put("Avg_Loss/Diffusion", avg_diffusion_loss)


            if total_counter % 5 == 0:
                step = total_counter // 5
                avgloss.to_tensorboard(writer, step)
                # Display the first image in the batch for the reconstruction comparison
                utils.tb_display_reconstruction_3D(writer, epoch, pet_images[0].detach().cpu(), out_ema[0], label="Sampled PET vs Original PET (EMA)")
                # Optional: log visual metrics
                with torch.no_grad():
                    psnr_val_recon = peak_signal_noise_ratio(pet_images[0].detach().cpu().numpy(), recon_pet[0].detach().cpu().numpy(), data_range=1.0)
                    ssim_val_recon = structural_similarity(pet_images[0][0].detach().cpu().numpy(), recon_pet[0][0].detach().cpu().numpy(), data_range=1.0)
                    # psnr_val_sample_raw = peak_signal_noise_ratio(pet_images[0].detach().cpu().numpy(), out_raw[0], data_range=1.0)
                    # ssim_val_sample_raw = structural_similarity(pet_images[0][0].detach().cpu().numpy(), out_raw[0][0], data_range=1.0)
                    psnr_val_sample_ema = peak_signal_noise_ratio(pet_images[0].detach().cpu().numpy(), out_ema[0], data_range=1.0)
                    ssim_val_sample_ema = structural_similarity(pet_images[0][0].detach().cpu().numpy(), out_ema[0][0], data_range=1.0)
                    writer.add_scalar("Metric/PSNR_Recon", psnr_val_recon, step)
                    writer.add_scalar("Metric/SSIM_Recon", ssim_val_recon, step)
                    writer.add_scalar("Metric/PSNR_Sample_EMA", psnr_val_sample_ema, step)
                    writer.add_scalar("Metric/SSIM_Sample_EMA", ssim_val_sample_ema, step)
 
        if (epoch + 1) % 50 == 0:
            save_checkpoint_BB(model, optimizer, epoch, checkpoint_dir, model_config, model_name="ct2pet_diffusion_model")
            if use_ema:
                torch.save(ema.state_dict(), os.path.join(checkpoint_dir, f"ema_model_epoch{epoch+1}.pt"))

# Main Entrypoint
def main(model_config, device):
    model = initialize_model(model_config, device).to(device)
    autoencoder_checkpoint_path = model_config.district_checkpoints[model_config.district_name]
    load_autoencoder_checkpoint(model.autoencoder, autoencoder_checkpoint_path, device)

    for name, param in model.named_parameters():
        if "autoencoder" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1.5e-5, eps=1e-6 if model_config.amp else 1e-8
    )
    # loss_handler = VAE_Losses(device, perceptual_weight=0.05, kl_weight=1e-7, adv_weight=0.1, contrastive_weight=0.5)

    print("[INFO] Loading dataset...")
    data_loader = CreateDataloader(model_config, shuffle=True)
    print("Steps per epoch:", len(data_loader))
    train_ct2pet_diffusion(model, model.autoencoder, optimizer, data_loader, device, model_config)

if __name__ == '__main__':
    model_config = ModelConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(model_config, device)
