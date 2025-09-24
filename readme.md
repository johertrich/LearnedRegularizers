# Learning Regularization Functionals: A Comparative Study

This repository contains the implementations for the chapter [Learned Regularization Functionals: A Comparative Study](https://arxiv.org/abs/xxxx.xxxx). If you have any questions or remarks, feel free to open an issue.

This `readme` file is structured into

1. Installation instructions of the dependencies
2. An overview of implemented regularizer architectures and instructions how to use them for reconstruction
3. An overview of implemented training methods
4. Instructions to reproduce the evaluation runs from the chapter
5. Instructions to reproduce the training runs from the chapter
6. Instructions to reproduce the automatic parameter fitting routine (Experiment 2) from the chapter
7. A `bibtex` citation for the chapter

## 1. Installation

To get started using `conda`, clone the repository and run (may take a few minutes)
```
conda env create --file=environment.yaml
```

In particular, the code relies on the [DeepInverse library](https://deepinv.github.io), which can be installed as follows:

```
pip install deepinv
```

## 2. Overview of Regularizers and Reconstruction

The implementation contains several regularizers (see table below) for which we provide trained weights. Creating a regularizer object consists out of three steps:

1. Create the regularizer itself with the import and constructor stated in the first table below.
2. For all architectures despite PatchNR, EPLL and LPN: use `from priors import ParameterLearningWrapper` and `regularizer = ParameterLearningWrapper(regularizer)` to incorporate learned regularization parameters
3. Load the weights with `regularizer.load_state_dict(torch.load(path))` with path stated in the paragraph "Weight Paths" below.

Then every regularizer has the fields `regularizer.g(x)` to evaluate the regularizer at `x` and `regularizer.grad(x)` to compute the gradient at `x`, where `x` is in the standard image format (batch size x channels x height x width).

Once the regularizer is loaded it can be used to solve an inverse problem using the nonmontonic accelerated (proximal) gradient descent (nmAPG) with the following code, where `physics` and `data_fidelity` provide the forward operator and data fidelity term in the format of the [DeepInverse library](https://deepinv.github.io):

```
from evaluation import reconstruct_nmAPG

# x_gt is the ground truth
y = physics(x_gt)  # y is the observation
lmbd = 1  # regularization parameter
step_size = 0.1  # initial step size in the nmAPG
maxiter = 1000  # maximal number of iterations in the nmAPG
tol = 1e-4  # stopping criterion
recon = reconstruct_nmAPG(y, physics, data_fidelity, regularizer, lmbd, step_size, maxiter, tol)
```

### Imports and Constructors

| Regularizer Name | Import | Constructor |
| ---------------- | ------ | ------------|
| CRR              | `from priors import WCRR` | `WCRR(sigma=0.1, weak_convexity=0.0)`|
| WCRR             | `from priors import WCRR` | `WCRR(sigma=0.1, weak_convexity=1.0)` |
| ICNN             | `from priors import ICNNPrior` | `ICNNPrior(in_channels=1, channels=32, kernel_size=5)` |
| IDCNN             | `from priors import IDCNNPrior` | `IDCNNPrior(in_channels=1, channels=32, kernel_size=5, act_name=act_name)` where `act_name = "elu"` for the CT-AR example and `act_name = "smoothed_relu"` otherwise|
| CNN (for AR/LAR) | `from priors import LocalAR` | `LocalAR(in_channels=1, pad=True, use_bias=True, n_patches=-1, output_factor=output_factor)` with `output_factor= n_pixels / 321**2` where `n_pixels` is the number of pixels in the image to which the regularizer is applied|
| CNN (for bilevel) | `from priors import LocalAR` | `LocalAR(in_channels=1, pad=True, use_bias=False, n_patches=-1, reduction="sum", output_factor=1 / 142 ** 2)`|
| TDV             | `from priors import TDV` | `TDV(in_channels=1, num_features=32, multiplier=1, num_mb=3, num_scales=3, zero_mean=True)` |
| LSR (despite NETT) | `from priors import LSR` | `LSR(nc=[32, 64, 128, 256], pretrained_denoiser=False, alpha=1.0, sigma=3e-2)` |
| LSR (for NETT)     | `from priors import NETT` | `NETT(in_channels=1, out_channels=1, hidden_channels=64, padding_mode="zeros")` |
| EPLL     | `from priors import EPLL` | see below |
| PatchNR     | `from priors import PatchNR` | see below |

For PatchNR and EPLL the weight files also contain information about the architecture. We refer to the script `variational_reconstruction.py` (line 146 to 178) for an example how to load them.

### Weight Paths

We provide the weights for all experiments done in the chapter. To load the weights use `torch.load(path)` with `path` defined below, where `problem="Denoising"` or `problem="CT"` and `regularizer_name` is a string which defines the regularizer as in the above table.

- bilevel-JFB: `path=f"weights/bilevel_{problem}/{regularizer_name}_JFB_for_{problem}.pt"`
- bilevel-IFT: `path=f"weights/bilevel_{problem}/{regularizer_name}_IFT_for_{problem}.pt"`
- MAID: `path=f"weights/bilevel_{problem}/{regularizer_name}_IFT-MAID_for_{problem}.pt"`
- AR/LAR: `path=f"weights/adversarial_{problem}/{regularizer_name}_for_{problem}_fitted.pt"`
- other: the PatchNR weights are in the directory `weights/patchnr`, the weights for EPLL are top-level in the `weights` directory. The LPN weights are in the directories `weights/lpn*`

## 3. Overview of Training Methods

We proivde generic scripts for the different training routines. These training routines include:

- Bilevel training with IFT/JFB: use `from training_methods import bilevel_training`, where the arguments are defined in the top of the script `training_methods/bilevel_training.py`
- Bilevel training with MAID: use `from training_methods import bilevel_training_maid`, where the arguments are defined in the top of the script `training_methods/bilevel_training_maid.py`
- (Local) Adversarial Regularization: use `from training_methods import ar_training`, where the arguments are defined in the top of the script `training_methods/ar_training.py`
- Score matching:  use `from training_methods import score_training`, where the arguments are defined in the top of the script `training_methods/score_training.py`
- Other: Similar there are the training routines for NETT, LPN, EPLL and PatchNR which are tailored to specific regularizers. The corresponding training scripts are top level (using `training_methods/nett_training.py` and `training_methods/lpn_training.py`)

## 4. Reproduce the Evaluation Runs (Experiment 1 and 3)

We describe how the learned regularizers can be evaluated (e.g. to solve the variational problem). The code for reproducing the baselines is located in the `baselines` directory.

The weights used for generating the numbers in the chapter are contained in the repository. Alternatively, they can be regenerated as described in part 5 of the `readme`.

### Variational Reconstruction (everything despite LPN)

The script `variational_reconstruction.py` provides a unified evaluation routine for most of the regularizers included. An example command is the following:

```
python variational_reconstruction.py --problem Denoising --evaluation_mode IFT --regularizer_name CRR
```

The arguments can be chosen as follows:
- `--problem` is set to `Denoising` for experiment 1 and to `CT` for experiment 3
- `--evaluation_mode` is set to `IFT` for bilevel-IFT, `JFB` for bilevel-JFB,  `IFT-MAID` for MAID, `Score` for score matching and `AR` for (local) adverserial regualrization
- `--regularizer_name` defines the regularizer architecture. Valid names are `CRR`, `WCRR`, `ICNN`, `IDCNN`, `LAR` (referring to the CNN), `TDV`, `LSR`, `PatchNR`, `EPLL` and `NETT`.

Other comments:
- for `EPLL` and `PatchNR` use `--evaluation_mode Score` (even though that's not quite accurate)
- for `NETT` use both `--evaluation_mode NETT` and `--regularizer_name NETT`
- If you want to save the first 10 reconstruction you can add the flag `--save_results True`

### Learned Proximal Networks

Even though the LPN provably defines a regularizer, evaluating it (or its gradient) requires to solve a (convex) optimization problem. Therefore, we evaluate the LPN in a Plug-and-Play fashion. In the denoising case this is a one-step reconstruction, in the CT case this is based on the ADMM algorithm.

To evaluate experiment 1 run:

```
python eval_LPN.py --problem Denoising --dataset BSD
```

To evaluate experiment 3 run:

```
python eval_LPN.py --problem CT --dataset LoDoPaB
```

## 5. Reproduce the Training Runs (Experiment 1 and 3)

To reproduce the training runs, we have unified scripts to reproduce all bilevel methods, all adversarial regularization methods and custom routines for NETT, EPLL, PatchNR and LPN,

### Bilevel Learning (BL-IFT, BL-JFB, MAID)

For reproducing the bilevel results, use the script `training_bilevel.py`, e.g., as follows
```
python training_bilevel.py --problem Denoising --hypergradient IFT --regularizer_name CRR
```
where the arguments can be chosen as
- `--problem` can be either `Denoising` (experiment 1) or `CT` (experiment 3)
- `--hypergradient` can be `IFT`, `JFB` or `IFT-MAID`
- `--regularizer_name` can be `CRR`, `WCRR`, `ICNN`, `IDCNN`, `LAR` (for the CNN column), `TDV` or `LSR`.


### Adversarial Regularization (AR/LAR)

For reproducing the AR/LAR runs, use the script `training_AR.py`, e.g., as follows
```
python training_AR.py --problem Denoising --regularizer_name CRR
```
where the arguments can be chosen as
- `--problem` can be either `Denoising` (experiment 1) or `CT` (experiment 3)
- `--regularizer_name` can be `CRR`, `WCRR`, `ICNN`, `IDCNN`, `LAR` (for the CNN column) or `TDV`.

### Custom Routines

For experiment 1 the EPLL, PatchNR, LPN and NETT can be trained by the commands
```
python training_EPLL.py --problem Denoising
python training_patchnr.py --problem Denoising
python training_LPN.py --dataset BSD
python training_nett.py --problem Denoising
```
For experiment 3, the commands are
```
python training_EPLL.py --problem CT
python training_patchnr.py --problem CT
python training_LPN.py --dataset LoDoPaB
python training_NETT.py --problem CT
```


## 6. Reproduce Denoising to CT (Experiment 2)

To reproduce the experiment 2 (training for denoising on BSDS and evaluating for CT on LoDoPaB), use the script `parameter_fitting_Denoising_to_CT.py`, e.g., by
```
python parameter_fitting_Denoising_to_CT.py --evaluation_mode bilevel-IFT --regularizer_name CRR
```
with arguments
- `--evaluation_mode` can be `bilevel-IFT`, `bilevel-JFB`, `AR`, `Score` or `NETT`. Use `Score` for EPLL and PatchNR (even though it's not quite accurate)
- `--regularizer_name` can be `CRR`, `WCRR`, `ICNN`, `IDCNN`, `LAR` (for the CNN column), `PatchNR`, `EPLL`, `TDV`, `LSR` or `NETT`

Again, the LPN requires a custom routine. As common for Plug-and-Play methods the noise level during training partially determines the regularization strength. Therefore, we retrain the regularizer on denoising with a smaller noise level and then evaluate it on CT by calling
```
python training_LPN.py --dataset BSD --noise_level 0.05
python eval_LPN.py --problem CT --dataset BSD
```

## 7. Citation

```
To be inserted
```
