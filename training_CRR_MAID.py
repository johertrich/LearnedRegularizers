#%%
from priors import WCRR
import torch
from deepinv.physics import Denoising
from training_methods import simple_bilevel_training_maid
from deepinv.optim import L2
from dataset import get_dataset
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from torchvision.transforms import RandomCrop, CenterCrop, Compose
import torchvision.transforms as transforms
import numpy as np
import random
from torchvision.transforms import (
    RandomCrop,
    RandomVerticalFlip,
    Compose,
    RandomHorizontalFlip,
    CenterCrop,
    RandomApply,
    RandomRotation,
)

# Fix random seed: optional, but recommended for reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
np.random.seed(0)
random.seed(0)


class GaussianNoise(torch.nn.Module):
    def __init__(self, sigma=0.1, rng: torch.Generator = torch.default_generator):
        super().__init__()
        self.sigma = sigma
        self.rng = rng
        self.noise = None
        self.noise_validation = None

    def forward(self, x):
        """
        Adds Gaussian noise to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Noisy tensor.
        """
        if self.noise == None:
            self.noise = torch.randn_like(x) * self.sigma
            self.noise = self.noise.cpu().detach()
            noise = self.noise.to(x.device)
        elif self.noise.shape != x.shape:
            if self.noise_validation == None:
                self.noise_validation = torch.randn_like(x) * self.sigma
                self.noise_validation = self.noise_validation.cpu().detach()
            noise = self.noise_validation.to(x.device)
        else:
            noise = self.noise.to(x.device)
        return x + noise


if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

problem = "Denoising"

# problem dependent parameters
noise_level = 0.1
physics = Denoising(noise_model=GaussianNoise(sigma=noise_level))
data_fidelity = L2(sigma=1.0)
lmbd = 1.0

# Define patch parameters
PATCH_SIZE = 64
STRIDE = 64  # Use PATCH_SIZE for non-overlapping patches
SUBSET = 32
DETERMINISTIC = True
CHANGE_DATASET = True  # Set to True if you want MAID to run on a different dataset after several epochs
NUM_REPS = 30  # Number of repetitions for running MAID on a different dataset

# --- Helper Function to Extract Patches Deterministically ---
def extract_patches(image_tensor, patch_size, stride):
    """
    Extracts patches from a single image tensor deterministically.

    Args:
        image_tensor (torch.Tensor): Input image tensor (C, H, W).
        patch_size (int): The height and width of the patches.
        stride (int): The step size between patches.

    Returns:
        list[torch.Tensor]: A list of patch tensors.
    """
    patches = []
    _, h, w = image_tensor.shape
    # Iterate over the image grid with the specified stride
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = image_tensor[:, i : i + patch_size, j : j + patch_size]
            patches.append(patch)
    return patches


# --- Custom Dataset for Patches ---
class PatchesDataset(Dataset):
    """
    A dataset that holds deterministic patches extracted from an original dataset.
    """

    def __init__(self, original_dataset, patch_size, stride, transform=None):
        """
        Args:
            original_dataset (Dataset): The original dataset (e.g., BSDS500).
            patch_size (int): The size of the patches to extract.
            stride (int): The stride for patch extraction.
            transform (callable, optional): Optional transform to be applied
                                             *after* patch extraction.
        """
        self.patch_size = patch_size
        self.stride = stride
        self.transform =  transform
        self.all_patches = []
        self.original_image_indices = []  # Optional: track origin

        print(f"Processing original dataset to extract patches...")
        pre_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply([transforms.RandomRotation(90)], p=0.5),
                transforms.ToTensor(),  # Convert PIL Image to [C, H, W] tensor [0,1]
            ]
        )

        num_original_images = len(original_dataset)
        for idx in range(num_original_images):
            original_image = original_dataset[idx]  # Get image, ignore label if any

            # Ensure it's a PIL image before applying pre_transform if needed
            if isinstance(original_image, Image.Image):
                image_tensor = pre_transform(original_image)
            else:
                image_tensor = original_image
            # Check if image is large enough for at least one patch
            c, h, w = image_tensor.shape
            if h < patch_size or w < patch_size:
                print(
                    f"Warning: Image {idx} (size {h}x{w}) is smaller than patch size {patch_size}x{patch_size}. Skipping."
                )
                continue

            # Extract patches for the current image
            patches = extract_patches(image_tensor, self.patch_size, self.stride)
            self.all_patches.extend(patches)

            # Optional: Store which original image each patch came from
            self.original_image_indices.extend([idx] * len(patches))

            if (idx + 1) % 50 == 0 or (idx + 1) == num_original_images:
                print(f"  Processed {idx + 1}/{num_original_images} original images...")

        print(f"Finished extracting patches. Total patches: {len(self.all_patches)}")

    def __len__(self):
        """Returns the total number of patches."""
        return len(self.all_patches)

    def __getitem__(self, idx):
        """Returns the patch at the given index."""
        patch = self.all_patches[idx]
        # Optional: get original image index: original_idx = self.original_image_indices[idx]

        if self.transform:
            patch = self.transform(patch)  # Apply any post-patching transforms

        # Return patch (and optionally original_idx if needed later)
        return patch  # , original_idx


transform = Compose(
    [
        RandomCrop(64),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomApply([RandomRotation((90, 90))], p=0.5),
    ]
)
train_dataset = get_dataset("BSDS500_gray", test=False)
val_dataset = get_dataset("BSD68", test=False, transform=CenterCrop(PATCH_SIZE))

patch_dataset = PatchesDataset(
    original_dataset=train_dataset,
    patch_size=PATCH_SIZE,
    stride=STRIDE,
    transform=None,  # Add post-patch transforms here if needed
)
num_patches_total = len(patch_dataset)
num_subset_patches = SUBSET
if num_patches_total == 0:
    print("No patches were extracted. Check dataset and parameters.")
    exit()

# Use a fixed range for deterministic subset selection
if DETERMINISTIC:
    subset_indices = torch.randint(0, num_patches_total, (num_subset_patches,))
    train_subset = Subset(patch_dataset, subset_indices)
else:
    train_set = get_dataset("BSDS500_gray", test=False, transform=transform)

VALIDATION_SIZE = 16  # Number of validation patches to use
val_set = torch.utils.data.Subset(val_dataset, range(VALIDATION_SIZE))

# create dataloaders
if DETERMINISTIC:
    train_dataloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=num_subset_patches,
        shuffle=False,
        drop_last=True,
        num_workers=8,
    )
else:
    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=32, shuffle=True, drop_last=True, num_workers=8
    )

val_dataloader = torch.utils.data.DataLoader(
    val_set, batch_size=8, shuffle=False, drop_last=True, num_workers=8
)

# define regularizer
weakly = False
regularizer = WCRR(
    sigma=0.1, weak_convexity=1.0 if weakly else 0.0, pretrained=None
).to(device)

# In case you want to experiment with different lower-level accuracy and upper-level step sizes
eps_list = [1e-1]
alpha_list = [1e-1]

for eps in eps_list:
    for alpha in alpha_list:
        print(f"Training with eps: {eps}, alpha: {alpha}")
        if CHANGE_DATASET:
            eps0 = eps
            alpha0 = alpha
            for rep in range(NUM_REPS):
                print(f"Repetition {rep + 1}/{NUM_REPS} for eps: {eps}, alpha: {alpha}")
                subset_indices = torch.randint(
                    0, num_patches_total, (num_subset_patches,)
                )
                train_subset = Subset(patch_dataset, subset_indices)
                train_dataloader = torch.utils.data.DataLoader(
                    train_subset,
                    batch_size=num_subset_patches,
                    shuffle=False,
                    drop_last=True,
                    num_workers=8,
                )
        
                if rep == 0:
                    logs = None  
                    optimizer = None 
                if rep == NUM_REPS - 1:
                    epochs = 200 # to ensure convergence in the last repetition
                else:
                    epochs = 50   
                if eps < 1e-4:
                    eps = 1e-4  # Ensure eps is not too small for stability          
                regularizer, loss_train, loss_val, psnr_train, psnr_val , eps, alpha, logs, _ = (
                    simple_bilevel_training_maid(
                        regularizer,
                        physics,
                        data_fidelity,
                        lmbd,
                        train_dataloader,
                        val_dataloader,
                        epochs= epochs,
                        NAG_step_size=1e-1,
                        NAG_max_iter=1000,
                        NAG_tol_train=eps,
                        NAG_tol_val=1e-4,
                        CG_tol=eps,
                        lr=alpha,
                        lr_decay=0.5,
                        device=device,
                        precondition=True,  # Use preconditioned upper-level optimization
                        verbose=False,
                        save_dir=str(SUBSET)
                        + "_"
                        + str(eps0)
                        + "_"
                        + str(alpha0)
                        + "_ICNN_Ada_chain",  # Directory to save the model and logs
                        # val_checkpoint=[],  # You can specify in which epoch to save the model, or None to save the best model based on validation loss
                        logs=logs,  # Dictionary to store logs
                        optimizer=optimizer,  # Optimizer to use for the training
                        algorithm="MAID Adagrad",  # Algorithm used for training
                    )
                )
                # controlling alpha in chain to not blow up
                if alpha > 0.5:
                    alpha = 0.5
                if alpha < 1e-5:
                    alpha = 1e-5
                physics = Denoising(noise_model=GaussianNoise(sigma=noise_level))

        else:
            print(f"Training on fixed dataset with eps: {eps}, alpha: {alpha}")
            regularizer, loss_train, loss_val, psnr_train, psnr_val, eps, alpha, logs, _ = (
                simple_bilevel_training_maid(
                    regularizer,
                    physics,
                    data_fidelity,
                    lmbd,
                    train_dataloader,
                    val_dataloader,
                    epochs=300,
                    NAG_step_size=1e-1,
                    NAG_max_iter=1000,
                    NAG_tol_train=eps,
                    NAG_tol_val=1e-4,
                    CG_tol=eps,
                    lr=alpha,
                    lr_decay=0.25,
                    device=device,
                    precondition=True,  # Use preconditioned upper-level optimization (AdaGrad)
                    verbose=True,
                    save_dir=str(SUBSET)
                    + "_"
                    + str(eps)
                    + "_"
                    + str(alpha)
                    + "_CRR_MAID",  # Directory to save the model and logs
                    algorithm = "MAID Adagrad",  # Algorithm used for training
                )
            )
        print(f"Training completed with eps: {eps}, alpha: {alpha}")
