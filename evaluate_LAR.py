"""
Evaluation Script for the Local CNN (trained with adversarial loss: Local AR)

@author: Alex
"""

from deepinv.physics import Denoising, MRI, GaussianNoise, Tomography
from deepinv.optim import L2, Tikhonov
from deepinv.utils.plotting import plot
from evaluation import evaluate
from dataset import get_dataset
from torchvision.transforms import CenterCrop, RandomCrop
import torch
import time
from operators import get_evaluation_setting

import os 
import yaml 

from priors import LocalAR


problem = "CT" # "CT"
#pretrained_weights = "weights/LocalAR_bilevel_JFB_p=15x15_LoDoPab.pt"
pretrained_weights = "weights/LocalAR_bilevel_IFT_p=15x15_BSD500.pt" #"weights/LocalAR_adversarial_p=15x15_BSD500.pt" # "weights/LocalAR_bilevel_IFT_p=15x15_BSD500.pt" #"LocalAR_adversarial_p=15x15_BSD500.pt"
# "LocalAR_{trainingmethod}_p={patchszie}_{training_data}.pt"

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("device: ", device)
torch.random.manual_seed(0)  # make results deterministic

############################################################
train_on = pretrained_weights.split("_")[-1].split(".")[0]

if "bilevel" in pretrained_weights:
    bilevel_training = True 
    bilevel_training_method = pretrained_weights.split("_")[2]
    print("Training method: ", bilevel_training_method)
else:
    bilevel_training = False

if bilevel_training:
    save_path = f"LAR/bilevel/{bilevel_training_method}/{problem}/train_on={train_on}"
else:
    save_path = f"LAR/adversarial/{problem}/train_on={train_on}"

os.makedirs(save_path, exist_ok=True)

save_path_imgs = os.path.join(save_path, "imgs")
os.makedirs(save_path_imgs, exist_ok=True)

# Problem selection

only_first = False  # just evaluate on the first image of the dataset for test purposes

############################################################

# reconstruction hyperparameters, might be problem dependent
if bilevel_training:
    if "ift" in bilevel_training_method.lower():
        if problem == "Denoising":
            lmbd = 28.0  
            NAG_step_size = 1e-1  # step size in NAG
            NAG_max_iter = 300  # maximum number of iterations in NAG
            NAG_tol = 1e-6  # tolerance for the relative error (stopping criterion)
        elif problem == "CT":
            if train_on == "BSD500":
                lmbd = 8500. 
                NAG_step_size = 1e-1  # step size in NAG
                NAG_max_iter = 300  # maximum number of iterations in NAG
                NAG_tol = 1e-4  # tolerance for the relative error (stopping criterion)
            elif train_on == "LoDoPab":
                lmbd = 1.0 # TODO
                NAG_step_size = 1e-2  # step size in NAG
                NAG_max_iter = 300 #300  # maximum number of iterations in NAG
                NAG_tol = 1e-6  # tolerance for the relative error (stopping criterion)
    elif "jfb" in bilevel_training_method.lower():
        if problem == "Denoising":
            lmbd = 30.0 # TODO
            NAG_step_size = 1e-1  # step size in NAG
            NAG_max_iter = 300 #300  # maximum number of iterations in NAG
            NAG_tol = 1e-6  # tolerance for the relative error (stopping criterion)
        elif problem == "CT":
            if train_on == "BSD500":
                lmbd = 1.0 # TODO
                NAG_step_size = 1e-2  # step size in NAG
                NAG_max_iter = 300 #300  # maximum number of iterations in NAG
                NAG_tol = 1e-6  # tolerance for the relative error (stopping criterion)
            elif train_on == "LoDoPab":
                lmbd = 1.2
                NAG_step_size = 1e-2  # step size in NAG
                NAG_max_iter = 300 #300  # maximum number of iterations in NAG
                NAG_tol = 1e-6  # tolerance for the relative error (stopping criterion)
else: # "adversarial training"
    if problem == "Denoising":
        lmbd = 1500.0 
        NAG_step_size = 1e-1  # step size in NAG
        NAG_max_iter = 300  # maximum number of iterations in NAG
        NAG_tol = 1e-6  # tolerance for the relative error (stopping criterion)

    elif problem == "CT":
        if train_on == "BSD500":
            lmbd = 62000.
            NAG_step_size = 1e-2  # step size in NAG
            NAG_max_iter = 300 #300  # maximum number of iterations in NAG
            NAG_tol = 1e-6  # tolerance for the relative error (stopping criterion)
        elif train_on == "LoDoPab":
            lmbd = 60000.
            NAG_step_size = 1e-2  # step size in NAG
            NAG_max_iter = 300 #300  # maximum number of iterations in NAG
            NAG_tol = 1e-4  # tolerance for the relative error (stopping criterion)
#############################################################
############# Problem setup and evaluation ##################
############# This should not be changed   ##################
#############################################################

dataset, physics, data_fidelity = get_evaluation_setting(problem, device)


# Call unified evaluation routine
# Define regularizer

regularizer = LocalAR(
    patch_size=15,
    n_patches=-1,
    in_channels=1,
    pretrained=pretrained_weights,
    pad=True,
    use_bias=False if bilevel_training else True 
)
regularizer.to(device)




start = time.time()
### Evauate using NAG
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
    adaptive_range=True if problem == "CT" else False,
    save_path=save_path_imgs,
    save_png=True
)


end = time.time()

results_dict = {"psnr": float(mean_psnr), 
                "time": end-start, "weights": pretrained_weights,
                "lambda": lmbd,
                "NAG_step_size": NAG_step_size,
                "NAG_max_iter":NAG_max_iter,
                "NAG_tol": NAG_tol}


print(f"TIME: {end-start}s")

print(results_dict)
weight_name = pretrained_weights.split(".")[0].split("/")[-1]
results_name = f"{save_path}/{weight_name}_{problem}_{lmbd}.yaml"
with open(os.path.join(results_name), "w") as f:
    yaml.dump(results_dict, f)
# plot ground truth, observation and reconstruction for the first image from the test dataset
#plot([x_out, physics.A_dagger(y_out), recon_out])
