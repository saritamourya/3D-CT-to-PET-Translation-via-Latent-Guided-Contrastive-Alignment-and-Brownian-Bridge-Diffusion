
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from monai.transforms import SpatialCropd
from monai.networks.nets import resnet
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from configs.test_options import TestOptions
from models.models import create_model
from data.data_CACHE_TEST import CreateDataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
patch_dim = 32
overlap_dim = 16

# === Utility Functions ===
def patch_indices(image_shape, patch_size=(patch_dim, patch_dim, patch_dim), overlap=(overlap_dim, overlap_dim, overlap_dim)):
    indices = []
    step = np.subtract(patch_size, overlap)
    for x in range(0, image_shape[0], step[0]):
        for y in range(0, image_shape[1], step[1]):
            for z in range(0, image_shape[2], step[2]):
                center = (
                    min(x + patch_size[0] // 2, image_shape[0] - overlap[0]),
                    min(y + patch_size[1] // 2, image_shape[1] - overlap[1]),
                    min(z + patch_size[2] // 2, image_shape[2] - overlap[2])
                )
                indices.append(center)
    return indices

def denorm(tensor, a_min=0, a_max=20):
    return tensor * (a_max - a_min) + a_min

def extract_volume_feature(volume_tensor, model):
    with torch.no_grad():
        volume_tensor = volume_tensor.to(next(model.parameters()).device)
        feature = model(volume_tensor)
        return feature.view(feature.size(0), -1).cpu().numpy().squeeze()

def compute_pet_metrics(suv_values):
    if len(suv_values) == 0:
        suv_values = np.array([0.0])

    SUV_max = np.max(suv_values)
    SUV_mean = np.mean(suv_values)
    MTV_15 = np.sum(suv_values >= 1.5)
    MTV_25 = np.sum(suv_values >= 2.5)
    TLG_15 = MTV_15 * SUV_mean
    TLG_25 = MTV_25 * SUV_mean

    return {
        'SUV_max': SUV_max,
        'SUV_mean': SUV_mean,
        'MTV_1.5': MTV_15,
        'TLG_1.5': TLG_15,
        'MTV_2.5': MTV_25,
        'TLG_2.5': TLG_25
    }

def save_views(volume, prefix, output_dir):
    volume = volume.squeeze().numpy()
    volume = np.clip(volume, 0, 20)
    D, H, W = volume.shape
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(volume[D // 2, :, :], cmap='gray'); axs[0].set_title("Axial")
    axs[1].imshow(volume[:, H // 2, :], cmap='gray'); axs[1].set_title("Coronal")
    axs[2].imshow(volume[:, :, W // 2], cmap='gray'); axs[2].set_title("Sagittal")
    for ax in axs: ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_views.png"))
    plt.close()

# === Configuration ===
opt = TestOptions().parse()
opt.nThreads = 1
opt.batchSize = 32
opt.serial_batches = True
opt.no_flip = True
opt.results_dir = "./Resultsharedlatent_100petresidual_sh"
results_dir = os.path.join(opt.results_dir, opt.test_district)
os.makedirs(results_dir, exist_ok=True)
csv_path = os.path.join(results_dir, "patient_metrics.csv")

# === Data ===
test_loader = CreateDataloader(opt, shuffle=False, cache=False)
dataset_size = len(test_loader.dataset)
print(f"Result saving directory: {results_dir}")
save_indices = np.linspace(0, dataset_size - 1, 10, dtype=int)
print(f'# Testing images = {dataset_size}, saving slices for patients: {save_indices.tolist()}')



# === Model ===
model = create_model(opt)
feature_model = resnet.resnet18(spatial_dims=3, n_input_channels=1, num_classes=128).to(device)
patient_counter = 0
# === Inference Loop ===
all_results = []

for i, data in enumerate(test_loader):
    A_batch_list, B_batch_list = data["A"], data["B"]
    batch_size = len(A_batch_list)

    for b in range(batch_size):
        A = A_batch_list[b].unsqueeze(0)
        B = B_batch_list[b].unsqueeze(0)
        shape = A.shape[2:]
        patches = patch_indices(shape)
        _, _, D, H, W = A.shape

        generated = np.zeros((D, H, W), dtype=np.float16)
        overlap = np.zeros((D, H, W), dtype=np.float16)

        for center in patches:
            crop = SpatialCropd(keys=["A", "B"], roi_center=center, roi_size=[patch_dim]*3)
            patch = crop({
                "A": A.as_tensor().permute(0, 2, 3, 4, 1),
                "B": B.as_tensor().permute(0, 2, 3, 4, 1)
            })
            patch_A = patch['A'].squeeze().unsqueeze(0).unsqueeze(0).to(device)
            patch_B = patch['B'].squeeze().unsqueeze(0).unsqueeze(0).to(device)

            model.set_input({'A': patch_A, 'B': patch_B})
            model.test()
            fake_B = torch.tensor(model.get_current_visuals()['fake_B'])

            fake_np = fake_B.detach().cpu().numpy().squeeze()

            cx, cy, cz = center
            sx = sy = sz = overlap_dim
            generated[cx-sx:cx+sx, cy-sy:cy+sy, cz-sz:cz+sz] += fake_np
            overlap[ cx-sx:cx+sx, cy-sy:cy+sy, cz-sz:cz+sz] += 1

        generated /= np.maximum(overlap, 1)
        fake_image = generated
        real = denorm(B).squeeze().numpy()
        fake = denorm(torch.tensor(generated)).numpy()

        if patient_counter in save_indices:
            patient_dir = os.path.join(results_dir, f"patient_{patient_counter+1}")
            os.makedirs(patient_dir, exist_ok=True)
            save_views(torch.tensor(fake), "fake", patient_dir)
            save_views(torch.tensor(real), "real", patient_dir)


        mae = np.mean(np.abs(B.cpu().numpy().squeeze() - fake_image.squeeze()))
        psnr_val = psnr(real, fake, data_range=20.0)
        ssim_val, _ = ssim(real, fake, full=True, data_range=20.0)

        real_feat = extract_volume_feature(torch.tensor(real).unsqueeze(0).unsqueeze(0).float().to(device), feature_model)
        fake_feat = extract_volume_feature(torch.tensor(fake).unsqueeze(0).unsqueeze(0).float().to(device), feature_model)
        fid = np.linalg.norm(real_feat - fake_feat)

        real_metrics = compute_pet_metrics(real.squeeze())
        fake_metrics = compute_pet_metrics(fake.squeeze())

        row = {
            'Patient_ID': patient_counter+1,
            'MAE': mae,
            'PSNR': psnr_val,
            'SSIM': ssim_val,
            'FID': fid,
            'REAL_SUV_max': real_metrics['SUV_max'],
            'REAL_SUV_mean': real_metrics['SUV_mean'],
            'REAL_MTV_1.5': real_metrics['MTV_1.5'],
            'REAL_TLG_1.5': real_metrics['TLG_1.5'],
            'REAL_MTV_2.5': real_metrics['MTV_2.5'],
            'REAL_TLG_2.5': real_metrics['TLG_2.5'],
            'GEN_SUV_max': fake_metrics['SUV_max'],
            'GEN_SUV_mean': fake_metrics['SUV_mean'],
            'GEN_MTV_1.5': fake_metrics['MTV_1.5'],
            'GEN_TLG_1.5': fake_metrics['TLG_1.5'],
            'GEN_MTV_2.5': fake_metrics['MTV_2.5'],
            'GEN_TLG_2.5': fake_metrics['TLG_2.5'],
        }

        df = pd.DataFrame([row])
        write_header = not os.path.exists(csv_path)
        df.to_csv(csv_path, mode='a', header=write_header, index=False)
        print(f"[INFO] Saved Patient {patient_counter+1} metrics to: {csv_path}")

        patient_counter += 1

# ==== Average Metrics ====
df_all = pd.read_csv(csv_path)
df_avg = df_all.mean(numeric_only=True).to_frame().T
df_avg.to_csv(os.path.join(results_dir, "average_metrics.csv"), index=False)
print("[INFO] Saved average metrics")
