import os
import gc
# Prevent CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler

from monai.networks.nets import PatchDiscriminator
from data.dataset_CACHE import CreateDataloader
from models.autoencoder_ctpet import Autoencoder
from configs.train_options import TrainOptions
from utils import utils
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

from utils.losses import VAE_Losses  # Import loss handling class
from utils.checkpoints_utils import save_checkpoint, load_checkpoint  # Import checkpoint utilities


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_latent_space(z_ct, z_pet, epoch, district, checkpoint_dir, writer=None):
    """
    Plot and save the latent space for CT and PET in district-specific folder.
    
    Args:
        z_ct (torch.Tensor): Latent space representation for CT.
        z_pet (torch.Tensor): Latent space representation for PET.
        epoch (int): Current epoch number.
        district (str): District name (e.g., 'lung', 'kidney').
        checkpoint_dir (str): Base directory to save the plot.
        writer (SummaryWriter, optional): Tensorboard writer to log images.
    """
    # Convert latent tensors to numpy
    z_ct_np = z_ct.detach().cpu().numpy().reshape(len(z_ct), -1)
    z_pet_np = z_pet.detach().cpu().numpy().reshape(len(z_pet), -1)

    # Combine for joint t-SNE embedding
    z_combined = np.concatenate([z_ct_np, z_pet_np])
    
    if np.isnan(z_combined).any():
        print(f"[WARNING] NaNs found in latent space at epoch {epoch} — skipping t-SNE plot.")
        return


    # Use safe perplexity for t-SNE
    perplexity_value = min(30, len(z_combined) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)
    z_embedded = tsne.fit_transform(z_combined)

    # Save plot in district-named folder
    district_folder = os.path.join(checkpoint_dir, district)
    os.makedirs(district_folder, exist_ok=True)

    # Plot the latent points
    plt.figure(figsize=(6, 5))
    plt.scatter(z_embedded[:len(z_ct), 0], z_embedded[:len(z_ct), 1], label="CT", alpha=0.6)
    plt.scatter(z_embedded[len(z_ct):, 0], z_embedded[len(z_ct):, 1], label="PET", alpha=0.6)

    # Draw dashed lines between corresponding CT–PET latent vectors
    for idx in range(len(z_ct)):
        ct_point = z_embedded[idx]
        pet_point = z_embedded[len(z_ct) + idx]
        plt.plot([ct_point[0], pet_point[0]], [ct_point[1], pet_point[1]], 'w--', alpha=0.3, linewidth=0.8)

    plt.legend()
    plt.title(f"Latent Space - Epoch {epoch}")
    plt.tight_layout()

    # Save and log the image
    plot_filename = os.path.join(district_folder, f"latent_space_epoch_{epoch}.png")
    plt.savefig(plot_filename)

    if writer:
        img = plt.imread(plot_filename)
        writer.add_image("Latent_Space", torch.tensor(img).permute(2, 0, 1), epoch)

    plt.close()


def train_autoencoder(opt):

    #Load Dataset
    """Train the AutoencoderCTPET model to encode CT and PET images into a shared latent space."""
    print("[INFO] Loading dataset...")
    train_loader = CreateDataloader(opt, shuffle=True)
    
    if train_loader is None:
        print("[ERROR] Dataset could not be loaded!")
        return

    print("[INFO] Initializing Autoencoder model...")


    num_channels = [32, 64, 128]  # Ensure this is correct
    norm_num_groups = 32

    # -------Initialize the AutoencoderCTPET model--------
    autoencoder = Autoencoder(
        spatial_dims= 3,
        in_channels= 1,
        out_channels= 1,
        num_res_blocks = [2, 2, 2],
        num_channels= num_channels,
        attention_levels= [False, False, False] ,
        latent_channels = 3,
        norm_num_groups = norm_num_groups,
        norm_eps = 1e-6,
        with_encoder_nonlocal_attn= False,
        with_decoder_nonlocal_attn = False,
        include_fc = False,
        use_combined_linear = False,
        use_flash_attention = False,
        use_checkpointing = False,
        use_convtranspose = False,
        norm_float16 = False,
        print_info = False,
        save_mem = True,
    ).to(device)


    # **Initialize the Discriminator**
    discriminator = PatchDiscriminator(
        spatial_dims=3, num_layers_d=3, channels=32,
        in_channels=1, out_channels=1, norm="INSTANCE"
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")  
        autoencoder = nn.DataParallel(autoencoder)
        discriminator = nn.DataParallel(discriminator)
    

    # Loss handler
    loss_handler = VAE_Losses(
        device, perceptual_weight=opt.perceptual_weight, 
        kl_weight=opt.kl_weight, adv_weight=opt.adv_weight
    )

    # **Define Optimizers & Learning Rate Schedulers**
    optimizer_g = optim.Adam(filter(lambda p: p.requires_grad, autoencoder.parameters()), lr=opt.lr, eps=1e-6 if opt.amp else 1e-8)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=opt.lr, eps=1e-6 if opt.amp else 1e-8)
    scheduler_g = lr_scheduler.LambdaLR(optimizer_g, lr_lambda=lambda epoch: 0.1 if epoch < 10 else 1.0)
    scheduler_d = lr_scheduler.LambdaLR(optimizer_d, lr_lambda=lambda epoch: 0.1 if epoch < 10 else 1.0)

    # **Setup AMP GradScaler**
    scaler_g = GradScaler(enabled=opt.amp)
    scaler_d = GradScaler(enabled=opt.amp)

    # **Initialize Tensorboard Logging**
    writer = SummaryWriter(comment="eventlog_for_vae_training")
    avgloss=utils.AverageLoss()
    total_counter = 0    # print(f"TensorBoard logging directory: {writer.log_dir}")
    checkpoint_dir = "pretrained/checkpointssharedlatent_50PETresidual_withoutcontrastiveloss"

    # **Load Checkpoints from checkpoint.py**
    start_epoch = load_checkpoint(autoencoder, optimizer_g, checkpoint_dir, opt, model_name="autoencoder")
    _ = load_checkpoint(discriminator, optimizer_d, checkpoint_dir, opt, model_name="discriminator")
    print(f"Resuming training from epoch {start_epoch}")


    # apply_gradient_checkpointing(autoencoder.encoder)
    # apply_gradient_checkpointing(autoencoder.decoder)
    print("[INFO] Starting training...")
    for epoch in range(start_epoch, opt.n_epochs):
        autoencoder.train()
        discriminator.train()
        total_loss = {"recon": 0, "kl": 0, "perceptual": 0, "adv": 0}

        for i, batch in enumerate(train_loader):
             # Free unused memory before processing a new batch
            torch.cuda.empty_cache()
            gc.collect()
            img_ct = batch['A'].to(device)
            img_pet = batch['B'].to(device)
            # img_ct = torch.empty(1, 1, 200, 200, 200).to(device)
            # img_pet = torch.empty(1, 1, 150, 150, 150).to(device)
            print(f"shape of ct image:{img_ct.shape}, shape of pet image: {img_pet.shape}")

            optimizer_g.zero_grad()
            optimizer_d.zero_grad()

            with autocast(enabled=True, dtype=torch.float16):
            # **Encode CT & PET images into shared latent space**
                z_ct, z_pet, mu_ct, log_var_ct, mu_pet, log_var_pet, recon_pet = autoencoder(img_ct, img_pet)
                # print(f"z_ct shape:{z_ct.shape}, z_pet shape: {z_pet.shape}, recon_pet: {recon_pet}")
                # **Train Discriminator First**
                logits_fake_pet = discriminator(recon_pet.detach())[-1]  # Detach to avoid generator gradients
                logits_real_pet = discriminator(img_pet.detach())[-1]

                loss_d_pet = (
                    loss_handler.adv_loss(logits_fake_pet, target_is_real=False, for_discriminator=True) +
                    loss_handler.adv_loss(logits_real_pet, target_is_real=True, for_discriminator=True)
                )
                print(f"loss_d_pet:{loss_d_pet}")

                # **Backpropagation for Discriminator**
                scaler_d.scale(loss_d_pet).backward()
                # **Gradient Accumulation for Discriminator**
                if (i + 1) % opt.gradient_accumulation_steps == 0:
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                    optimizer_d.zero_grad()  # Reset gradients

                # **Compute Generator Losses**
                losses, loss_g = loss_handler.compute_losses(
                    z_ct, z_pet, recon_pet, img_pet, mu_ct, log_var_ct, mu_pet, log_var_pet, discriminator
                )
                
                print("DEBUG: Losses computed successfully!")  # Debugging line
                print(f"losses: {losses}, loss_g: {loss_g}")

                print(f"Recon PET: {recon_pet.shape}")
                print(f"Latent Mean: {mu_ct.shape}, Latent Mean: {mu_pet.shape}, Log Variance: {log_var_ct.shape}, Log Variance: {log_var_pet.shape}")

                # **Backpropagation for Generator**
                scaler_g.scale(loss_g).backward()

                # **Gradient Accumulation for Generator**
                if (i + 1) % opt.gradient_accumulation_steps == 0:
                    scaler_g.step(optimizer_g)
                    scaler_g.update()
                    optimizer_g.zero_grad()  # Reset gradients


                #---------Store losses-------
            # Track average loss
            avgloss.put("Loss/Reconstruction_PET", losses["recon"].item())
            avgloss.put("Loss/KL", losses["kl"].item())
            avgloss.put("Loss/Perceptual", losses["perceptual"].item())
            # avgloss.put("Loss/Contrastive", losses["contrastive"].item())
            avgloss.put("Loss/Adversarial", loss_d_pet.item())

                        # Track total loss for per-epoch logging
            for key in total_loss:
                if key in losses:
                    total_loss[key] += losses[key].item()
                if key == "adv":
                    total_loss["adv"] += loss_d_pet.item()

            # Visualize every N steps
            total_counter += 1
            if total_counter % 50 == 0:
                step = total_counter // 50
                avgloss.to_tensorboard(writer, step)
                utils.tb_display_reconstruction_3D(
                    writer, step,
                    img_pet[0].detach().cpu(),
                    recon_pet[0].detach().cpu(),
                    label="PET_Recon"
                )
            
        #writer.flush()
         # Step the learning rate scheduler after each epoch
        scheduler_g.step()
        scheduler_d.step()
        # Plot latent space every 50 epochs
        if (epoch + 1) % 50 == 0:
            if not (torch.isnan(z_ct).any() or torch.isnan(z_pet).any()):
                plot_latent_space(z_ct, z_pet, epoch + 1, opt.district, checkpoint_dir=checkpoint_dir, writer=writer)
            else:
                print(f"[WARNING] Skipping latent space plot at epoch {epoch + 1} due to NaNs in z_ct or z_pet.")

        
        # Epoch-average loss logging
        avg_loss = {k: v / len(train_loader) for k, v in total_loss.items()}
        for key, val in avg_loss.items():
            writer.add_scalar(f"Epoch_Avg/{key}", val, epoch + 1)

        # Optional: log learning rate
        writer.add_scalar("LR/Generator", scheduler_g.get_last_lr()[0], epoch + 1)
        print(f"Epoch [{epoch+1}/{opt.n_epochs}], Recon PET: {avg_loss['recon']:.6f}, KL Loss: {avg_loss['kl']:.6f}, Perceptual Loss: {avg_loss['perceptual']:.6f}, Adv Loss: {avg_loss['adv']:.6f}")

        # **Save Model Checkpoint**
        if (epoch + 1) % opt.val_interval == 0:
            save_checkpoint(autoencoder, optimizer_g, epoch, checkpoint_dir, opt, model_name="autoencoder")
            save_checkpoint(discriminator, optimizer_d, epoch, checkpoint_dir, opt,model_name="discriminator")

    print("[INFO] Training Complete! Model saved to pretrained/checkpoints")


if __name__ == "__main__":
    opt = TrainOptions()
    train_autoencoder(opt),