# Learned Variational Regularization: A Comparative Study 

The code relies on the [DeepInverse package](https://deepinv.github.io), which can be installed as follows:

```
pip install deepinv
```
Overall to get started using conda, clone the repository and run (may take a few minutes)
```
conda env create --file=environment.yaml
```

## Naming and Structure of Training and Evaluation Scripts

### Naming

Each training script for denoising should be named by `training_ARCHITECTURE_ALGORITHM.py` specifying architecture and training algorithm. For the CT example the naming should be the same with appending `_CT` at the end. The evaluation scripts should be named `example_ARCHITECTURE_ALGORITHM(_CT).py`. We might replace the part `example` by `inference` at some point, but I think it currently does not matter too much as long as we can see architecture 

### Loading the forward operator and evaluation setting

In order to ensure that all methods work in the same setting please use for training the forward model (`physics` which contains both, the forward operator `physics.A` and its adjoint `physics.A_adjoint`, and the noise model such that `physics(x)` applies the forward operator and the noise level to `x`), dataset and data fidelity term, please load them as follows:

For training:
```
from operators import get_operator
from dataset import get_dataset

problem = "Denoising" # or "CT"
device = "cuda" # or whatever is used
physics, data_fidelity = get_operator(problem, device)

transform = None  # feel free to use transforms for data augmentation etc.
if problem == "Denoising":
	train_dataset = get_dataset("BSDS500_gray", test=False, transform=transform) 
elif problem == "CT":
	train_dataset = get_dataset("LoDoPaB", test=False, transform=transform) 
```

For evaluation:
```
from operators import get_evaluation_setting

dataset, physics, data_fidelity = get_evaluation_setting(problem, device)
```

The regularizers should be evaluated by the prescribed evaluation routine, see below. The function takes the argument `adaptive_range` which determines whether we evaluate the PSNR using a fixed range or a dynamic range (determined by the maximal pixel value in the ground truth image). Please use `adaptive_range=True` for CT and `adaptive_range=False` for denoising:
```
from evaluation import evaluate

problem = "Denoising"
adaptive_range = True if problem == "CT" else False

mean_psnr, x_out, y_out, recon_out = evaluate(
    physics=physics,
    data_fidelity=data_fidelity,
    dataset=dataset,
    regularizer=regularizer,
    lmbd=lmbd,
    NAG_step_size=NAG_step_size,
    NAG_max_iter=NAG_max_iter,
    NAG_tol=NAG_tol,
    device=device,
    adaptive_range=adaptive_range,
    verbose=True,
)
```

### Placement in the repo

Work in progress scripts can be kept top-level. Once you are happy with the results please move the training and evaluation script to the directories `scripts_denoising_final` or `scripts_CT_final`.

### Example for a training script

The training script should have the same structure as the following example:
```
####################################
# Imports and device specification
####################################
from operators import get_operator
import torch
from training_methods import bilevel_training
from dataset import get_dataset

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

####################################
# Problem specification
####################################
problem = "Denoising" # "CT" for the CT problem

# call unified definition of the forward operator
physics, data_fidelity = get_operator(problem, device)


####################################
# Load dataset and dataloader
####################################

# The dataset should always be loaded via the get_dataset function,
# but you might define customized transforms (crops, data augmentation etc.) and  customized
# training/validation splits
transform = CenterCrop(321)
# "BSDS500_gray" for Denoising, "LoDoPaB" for CT
train_dataset = get_dataset("BSDS500_gray", test=False, transform=transform) 
val_dataset = get_dataset("BSDS500_gray", test=False, transform=transform)
# splitting in training and validation set
test_ratio = 0.1
test_len = int(len(train_dataset) * 0.1)
train_len = len(train_dataset) - test_len
train_set = torch.utils.data.Subset(train_dataset, range(train_len))
val_set = torch.utils.data.Subset(val_dataset, range(train_len, len(train_dataset)))

# create dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_set, batch_size=32, shuffle=True, drop_last=True, num_workers=8
)
val_dataloader = torch.utils.data.DataLoader(
    val_set, batch_size=1, shuffle=False, drop_last=True, num_workers=8
)

####################################
# define regularizer and parameters
####################################

noise_level = 0.1
lmbd = 1.0
regularizer = wcrr.WCRR(
    sigma=noise_level,
    weak_convexity=0.0,
).to(device)

####################################
# Call training script
####################################

regularizer, loss_train, loss_val, psnr_train, psnr_val = bilevel_training(
    regularizer,
    physics,
    data_fidelity,
    lmbd,
    train_dataloader,
    val_dataloader,
    epochs=200,
    NAG_step_size=1e-1,
    NAG_max_iter=1000,
    NAG_tol_train=1e-4,
    NAG_tol_val=1e-4,
    lr=0.01,
    lr_decay=0.99,
    device=device,
    verbose=False,
)

####################################
# save weights
####################################

torch.save(regularizer.state_dict(), f"weights/CRR_bilevel.pt")
```

### Example for a test script

The evaluation scripts should have the same structure as the following example:
```
####################################
# Imports and device and seed specification
####################################
from operators import get_evaluation_setting
from deepinv.utils.plotting import plot
from evaluation import evaluate
from dataset import get_dataset
from priors import WCRR
import torch

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

torch.random.manual_seed(0)  # make results deterministic

############################################################

# Problem selection

problem = "Denoising"  # Select problem setups, which we consider.
only_first = False  # just evaluate on the first image of the dataset for test purposes

############################################################

# Define regularizer and parameters

weakly = True
pretrained = "weights/WCRR_bilevel.pt" if weakly else "weights/CRR_bilevel.pt"
regularizer = WCRR(
    sigma=0.1, weak_convexity=1.0 if weakly else 0.0, pretrained=pretrained
).to(device)

# Parameters for the Nesterov Algorithm, might also be problem dependent...

NAG_step_size = 1e-1  # step size in NAG
NAG_max_iter = 1000  # maximum number of iterations in NAG
NAG_tol = 1e-4  # tolerance for the relative error (stopping criterion)


#############################################################
############# Problem setup and evaluation ##################
############# This should not be changed   ##################
#############################################################

# Define forward operator
dataset, physics, data_fidelity = get_evaluation_setting(problem, device)

# Call unified evaluation routine

mean_psnr, x_out, y_out, recon_out = evaluate(
    physics=physics,
    data_fidelity=data_fidelity,
    dataset=dataset,
    regularizer=regularizer,
    lmbd=lmbd,
    NAG_step_size=NAG_step_size,
    NAG_max_iter=NAG_max_iter,
    NAG_tol=NAG_tol,
    only_first=only_first,
    device=device,
    verbose=True,
)

# plot ground truth, observation and reconstruction for the first image from the test dataset
plot([x_out, y_out, recon_out])

```


