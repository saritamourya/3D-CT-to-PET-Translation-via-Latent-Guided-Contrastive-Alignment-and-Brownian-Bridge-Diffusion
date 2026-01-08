import torch.nn as nn
import torch.nn.functional as F
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
import torch

from models.utils import KL_loss

class VAE_Losses:
    """Class to encapsulate different loss functions for training AutoencoderCTPET."""

    def __init__(self, device, perceptual_weight=0.3, kl_weight=1e-7, adv_weight=0.1, contrastive_weight=0.1):
        """
        Initialize loss functions.

        Args:
            device: GPU or CPU device
            perceptual_weight: Weight for perceptual loss
            kl_weight: Weight for KL divergence loss
            adv_weight: Weight for adversarial loss (latent space regularization)
            contrastive_weight: Weight for contrastive loss (to align CT & PET while preserving PET-specific features)
        """
        self.recon_loss = nn.L1Loss()  # Reconstruction loss (L1)
        self.kl_loss = KL_loss  # KL divergence loss
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")
        self.perceptual_loss = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2, pretrained="DEFAULT").to(device)
        self.contrastive_weight = contrastive_weight


        self.perceptual_weight = perceptual_weight
        self.kl_weight = kl_weight
        self.adv_weight = adv_weight


    def contrastive_loss(self, z_ct, z_pet, temperature=0.1):
        """
        Encourages CT and PET embeddings to be aligned but distinct.
        Aims to retain PET-specific metabolic details while keeping structure aligned.
        """
        similarity = F.cosine_similarity(z_ct.view(z_ct.size(0), -1), z_pet.view(z_pet.size(0), -1), dim=-1)
        return -torch.mean(torch.log(torch.exp(similarity / temperature) / (torch.exp(similarity / temperature).sum())))

    def compute_losses(self, z_ct, z_pet, recon_pet, pet_image, mu_ct, log_var_ct, mu_pet, log_var_pet, discriminator):
        """
        Computes all necessary losses for encoding CT & PET into a shared latent space.

        Args:
            z_ct: Latent representation of CT
            z_pet: Latent representation of PET
            mu_ct: Mean of latent distribution (CT)
            log_var_ct: Log variance of latent distribution (CT)
            mu_pet: Mean of latent distribution (PET)
            log_var_pet: Log variance of latent distribution (PET)
            discriminator: Adversarial discriminator for latent space regularization
        """
        losses = {
            "recon": self.recon_loss(recon_pet, pet_image),  # Latent consistency loss
            "kl": self.kl_loss(mu_ct, log_var_ct) + self.kl_loss(mu_pet, log_var_pet),  # KL loss
            "perceptual": self.perceptual_loss(recon_pet, pet_image),  # Perceptual loss
            "contrastive": self.contrastive_loss(z_ct, z_pet), #encourage pet-specific preservation
        #    "latent_orthogonality": self.latent_orthogonality_loss(z_ct, z_pet)
        }

        # Adversarial loss: Discourage trivial latent encodings by making z_ct and z_pet realistic
        logits_fake_pet = discriminator(recon_pet.float())[-1]
        generator_loss_pet = self.adv_loss(logits_fake_pet, target_is_real=True, for_discriminator=False)

         # Total adversarial loss
        generator_loss = ( generator_loss_pet) * 0.5

        # Final weighted loss for the encoder
        loss_g = (
            losses["recon"]+ 
            self.kl_weight * losses["kl"] +  # Ensure smooth latent representation
            self.perceptual_weight * losses["perceptual"] +  # Maintain high-level structure
            self.adv_weight * generator_loss+# Adversarial regularization
            self.contrastive_weight*losses["contrastive"] #contrastive loss for structural alignement
        )

        return losses, loss_g
