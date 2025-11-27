import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
def find_label_center_loc(x):
    """
    Find the center location of non-zero elements in a binary mask.

    Args:
        x (torch.Tensor): Binary mask tensor. Expected shape: [H, W, D] or [C, H, W, D].

    Returns:
        list: Center locations for each dimension. Each element is either
              the middle index of non-zero locations or None if no non-zero elements exist.
    """
    label_loc = torch.where(x != 0)
    center_loc = []
    for loc in label_loc:
        unique_loc = torch.unique(loc)
        if len(unique_loc) == 0:
            center_loc.append(None)
        else:
            center_loc.append(unique_loc[len(unique_loc) // 2])

    return center_loc
def normalize_ct_pet(image, modality="CT"):
    """
    Normalize a 3D CT or PET image to [0,1] range for input into the autoencoder.

    Args:
        image (torch.Tensor): Input CT or PET image tensor. Expected shape: [C, H, W, D].
        modality (str): Either "CT" or "PET" to apply specific normalization.

    Returns:
        torch.Tensor: Normalized image.
    """
    if modality == "CT":
        # Clip HU values to focus on soft tissue (-1000 to 1000 is common)
        image = torch.clip(image, -1000, 1000)
        image = (image + 1000) / 2000  # Normalize to [0,1]
    elif modality == "PET":
        # PET values are highly variable, normalize based on percentile scaling
        min_val = image.min()
        max_val = image.max()
        image = (image - min_val) / (max_val - min_val + 1e-5)  # Avoid division by zero

    return image
def normalize_ct_pet_to_uint8(image, modality="CT"):
    """
    Convert CT or PET image to uint8 format for visualization.

    Args:
        image (torch.Tensor): Input CT or PET image tensor. Expected shape: [C, H, W, D].
        modality (str): Either "CT" or "PET" to apply specific normalization.

    Returns:
        numpy.ndarray: uint8 formatted image.
    """
    image = normalize_ct_pet(image, modality)  # Apply min-max normalization
    image = (image * 255).astype(np.uint8)  # Convert to uint8 range

    return image

def visualize_one_slice_in_3d(
    image: torch.Tensor,
    axis: int = 2,
    center: int = None,
    mask_bool: bool = True,
    n_label: int = 105,
    colorize: torch.Tensor = None,
    modality: str = "CT"
):
    """
    Extract and visualize a 2D slice from a 3D medical image or label tensor.

    Args:
        image (torch.Tensor): Input 3D image or label tensor. Shape: [C, H, W, D] (single-channel images: [1, H, W, D]).
        axis (int, optional): Axis along which to extract the slice (0, 1, or 2). Defaults to 2 (axial slice).
        center (int, optional): Index of the slice to extract. If None, the middle slice is used.
        mask_bool (bool, optional): If True, treats the input as a label mask and normalizes it. Defaults to True.
        n_label (int, optional): Number of labels in the mask. Used only if mask_bool is True. Defaults to 105.
        colorize (torch.Tensor, optional): Colorization weights for label normalization (for segmentation masks).
                                           Expected shape: [3, n_label, 1, 1] if provided.
        modality (str, optional): "CT" or "PET" to handle different image intensities. Defaults to "CT".

    Returns:
        numpy.ndarray: A 2D slice of the input image. If mask_bool is True, returns a colorized uint8 image [H, W, 3].
                       If mask_bool is False, returns a grayscale float32 array [H, W].
    """
    # Ensure the image is 4D: [C, H, W, D]
    if len(image.shape) != 4:
        raise ValueError(f"Expected image shape [C, H, W, D], but got {image.shape}")

    # If center is not given, select the middle slice
    if center is None:
        center = image.shape[axis + 1] // 2

    # Extract the slice along the given axis
    if axis == 0:  # Coronal view
        draw_img = image[:, center, :, :]
    elif axis == 1:  # Sagittal view
        draw_img = image[:, :, center, :]
    elif axis == 2:  # Axial (default) view
        draw_img = image[:, :, :, center]
    else:
        raise ValueError("axis should be in [0,1,2]")

    # If it's a segmentation mask, apply color mapping
    if mask_bool:
        draw_img = normalize_ct_pet_to_uint8(colorize, draw_img, n_label)

    else:  # If it's a CT or PET scan, apply normalization
        draw_img = draw_img.squeeze().cpu().numpy().astype(np.float32)
        draw_img = normalize_ct_pet(draw_img, modality)  # Normalize CT or PET
        draw_img = (draw_img * 255).astype(np.uint8)  # Convert to 8-bit

        # Convert grayscale image to 3-channel for visualization
        draw_img = np.stack((draw_img,) * 3, axis=-1)

    return draw_img
import numpy as np
import matplotlib.pyplot as plt

def show_image(image, title="Image", modality="CT", cmap=None):
    """
    Display a 2D or 3D slice of an input image.

    Args:
        image (numpy.ndarray or torch.Tensor): Image to be displayed.
            - Expected 2D shape: [H, W] (grayscale) or [H, W, 3] (RGB).
            - Expected 3D shape: [H, W, D].
        title (str, optional): Title for the plot. Defaults to "Image".
        modality (str, optional): "CT", "PET", or "segmentation" for appropriate scaling. Defaults to "CT".
        cmap (str, optional): Colormap for visualization (default: "gray" for CT, "hot" for PET).

    Returns:
        None (Displays the image using Matplotlib).
    """
    # Convert Torch tensor to NumPy if necessary
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    # Handle 3D input: Automatically selects a middle slice
    if len(image.shape) == 3:  # [H, W, D]
        center = image.shape[2] // 2
        image = image[:, :, center]  # Take the middle slice

    # Set colormap based on modality
    if modality.lower() == "ct":
        cmap = "gray" if cmap is None else cmap
        image = np.clip(image, -1000, 1000)  # Clip Hounsfield Units
        image = (image + 1000) / 2000  # Normalize to [0,1]
    elif modality.lower() == "pet":
        cmap = "hot" if cmap is None else cmap
        image = (image - image.min()) / (image.max() - image.min() + 1e-5)  # Normalize PET
    elif modality.lower() == "segmentation":
        cmap = None  # Segmentation masks should be RGB
    else:
        raise ValueError("Unknown modality. Choose from 'CT', 'PET', or 'segmentation'.")

    # # Convert grayscale images to RGB
    # if len(image.shape) == 2 and modality != "segmentation":
    #     image = (image * 255).astype(np.uint8)
    #     image = np.stack((image,) * 3, axis=-1)  # Convert grayscale to 3-channel RGB

    # Plot the image
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()
import numpy as np
import torch

def to_shape(a, shape, mode="constant", crop_if_larger=True):
    """
    Resize a 3D image to a desired shape by padding or cropping.

    Args:
        a (numpy.ndarray or torch.Tensor): Input 3D array/tensor to be resized. Shape: [X, Y, Z].
        shape (tuple): Desired output shape as (x_, y_, z_).
        mode (str, optional): Padding mode, default is "constant" (zero-padding).
                              Other modes: "edge", "reflect", etc. (used in np.pad).
        crop_if_larger (bool, optional): If True, crops the input if it is larger than the target shape.

    Returns:
        numpy.ndarray or torch.Tensor: Resized 3D image with shape [x_, y_, z_].

    Raises:
        ValueError: If the input is smaller than the desired shape and mode is invalid.

    Notes:
        - If input shape is larger than the target shape and `crop_if_larger=True`, cropping is applied.
        - If input shape is smaller, zero-padding (or specified mode) is applied symmetrically.
        - Works with both NumPy and PyTorch Tensors.
    """
    is_tensor = isinstance(a, torch.Tensor)  # Check if input is a PyTorch tensor
    if is_tensor:
        a = a.cpu().numpy()  # Convert to NumPy for padding/cropping

    x_, y_, z_ = shape  # Target shape
    x, y, z = a.shape   # Original shape

    # Handle cropping if input is larger than the target shape
    if crop_if_larger:
        if x > x_:
            crop_x = (x - x_) // 2
            a = a[crop_x: crop_x + x_, :, :]
        if y > y_:
            crop_y = (y - y_) // 2
            a = a[:, crop_y: crop_y + y_, :]
        if z > z_:
            crop_z = (z - z_) // 2
            a = a[:, :, crop_z: crop_z + z_]

    # Calculate padding values (ensure padding is symmetric)
    x_pad = max(0, x_ - a.shape[0])
    y_pad = max(0, y_ - a.shape[1])
    z_pad = max(0, z_ - a.shape[2])

    # Apply padding if necessary
    if x_pad > 0 or y_pad > 0 or z_pad > 0:
        a = np.pad(
            a,
            (
                (x_pad // 2, x_pad // 2 + x_pad % 2),
                (y_pad // 2, y_pad // 2 + y_pad % 2),
                (z_pad // 2, z_pad // 2 + z_pad % 2),
            ),
            mode=mode,
        )

    # Convert back to PyTorch tensor if original input was a tensor
    if is_tensor:
        a = torch.from_numpy(a)

    return a

def get_xyz_plot(image, center_loc_axis, mask_bool=True, n_label=105, colorize=None):
    """
    Generate a concatenated XYZ plot of 2D slices from a 3D medical image.

    This function extracts three orthogonal 2D slices (XY, XZ, YZ) from a 3D image 
    and combines them into a single visualization.

    Args:
        image (torch.Tensor or np.ndarray): Input 3D image. Shape: [C, H, W, D].
        center_loc_axis (list): List of three integers specifying slice indices along each axis.
        mask_bool (bool, optional): If True, applies mask normalization. Defaults to True.
        n_label (int, optional): Number of labels for visualization (used for masks). Defaults to 105.
        colorize (torch.Tensor, optional): Colorization weights. Shape: [3, n_label, 1, 1].

    Returns:
        numpy.ndarray: Concatenated 2D image of three orthogonal slices. Shape: [max(H,W,D), 3*max(H,W,D), 3].

    Notes:
        - Uses `to_shape()` to standardize output shape for visualization.
        - Can handle both PyTorch tensors & NumPy arrays.
        - Returns a single 2D RGB image containing three orthogonal views.
    """
    # Ensure input is a NumPy array for visualization
    is_tensor = isinstance(image, torch.Tensor)
    if is_tensor:
        image = image.cpu().numpy()  # Convert PyTorch tensor to NumPy

    assert image.ndim == 4, f"Expected input shape [C, H, W, D], got {image.shape}"

    target_shape = list(image.shape[1:])  # [H, W, D]
    img_list = []

    # Extract three orthogonal slices
    for axis in range(3):
        center = center_loc_axis[axis]

        img, _ = visualize_one_slice_in_3d(
            torch.tensor(image).unsqueeze(0) if is_tensor else image,  
            axis=axis,
            center=center,
            mask_bool=mask_bool,
            n_label=n_label,
            colorize=colorize,
        )

        # Ensure shape consistency across slices
        img = img.transpose([2, 1, 0])  # Ensure correct channel order
        img = to_shape(img, (3, max(target_shape), max(target_shape)))  # Standardize size

        img_list.append(img)

    # Concatenate three views horizontally
    final_img = np.concatenate(img_list, axis=2).transpose([1, 2, 0])

    return final_img
