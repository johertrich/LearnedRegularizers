from priors import ParameterLearningWrapper
from training_methods import bilevel_training
import torch
from operators import get_evaluation_setting
from dataset import get_dataset
from evaluation import evaluate

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

problem = "CT"

reg_name = "ICNN_bilevel"

if reg_name=="WCRR_bilevel":
    from priors import WCRR
    pretrained = "weights/WCRR_bilevel.pt"
    regularizer = WCRR(sigma=0.1, weak_convexity=1.0, pretrained=pretrained).to(device)
elif reg_name=="ICNN_bilevel":
    from priors import simple_ICNNPrior
    regularizer = simple_ICNNPrior(in_channels=1,channels=32,device=device)
    ckp = torch.load('weights/ICNN_bilevel.pt')
    regularizer.load_state_dict(ckp)
elif reg_name=="CRR_bilevel":
    from priors import WCRR
    pretrained = "weights/CRR_bilevel.pt"
    regularizer = WCRR(sigma=0.1, weak_convexity=0.0, pretrained=pretrained).to(device)
elif reg_name == "LSR_jfb":
    from priors import LSR
    pretrained = "weights/LSR_jfb.pt"
    regularizer = LSR(nc=[32, 64, 128, 256], pretrained_denoiser=False).to(device)
    regularizer.load_state_dict(torch.load(pretrained))
else:
    raise NameError("Unknown regularizer!")


lmbd_initial_guess = 60

dataset, physics, data_fidelity = get_evaluation_setting(problem, device)

validation_dataset = get_dataset("LoDoPaB_val")
validation_dataloader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=5, shuffle=False, drop_last=False, num_workers=8
)

wrapped_regularizer = ParameterLearningWrapper(regularizer, device=device)

for p in wrapped_regularizer.parameters():
    p.requires_grad_(False)
wrapped_regularizer.alpha.requires_grad_(True)
wrapped_regularizer.scale.requires_grad_(True)

# parameter search
wrapped_regularizer, loss_train, loss_val, psnr_train, psnr_val = bilevel_training(
    wrapped_regularizer,
    physics,
    data_fidelity,
    lmbd_initial_guess,
    validation_dataloader,
    validation_dataloader,
    epochs=100,
    mode="IFT",
    NAG_step_size=1e-1,
    NAG_max_iter=1000,
    NAG_tol_train=1e-4,
    NAG_tol_val=1e-4,
    lr=0.1,
    lr_decay=0.999,
    device=device,
    verbose=False,
    validation_epochs=100,
    dynamic_range_psnr=True,
)

only_first = False
wrapped_regularizer.alpha.requires_grad_(False)
wrapped_regularizer.scale.requires_grad_(False)
torch.random.manual_seed(0)  # make results deterministic

mean_psnr, x_out, y_out, recon_out = evaluate(
    physics=physics,
    data_fidelity=data_fidelity,
    dataset=dataset,
    regularizer=wrapped_regularizer,
    lmbd=lmbd_initial_guess,
    NAG_step_size=1e-2,
    NAG_max_iter=1000,
    NAG_tol=1e-4,
    only_first=only_first,
    device=device,
    verbose=True,
    adaptive_range=True,
)
