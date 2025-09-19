## No-Patch LPN

### Experiment 2: CT reconstruction using model trained on BSD.
Train:
```
python training_LPN.py --dataset BSD --ckpt_dir weights/lpn_no_patch_64_bsd_noise_0.09 --noise_level 0.09
```
Weights will be saved in `weights/lpn_no_patch_64_bsd_noise_0.09/LPN.pt`.
Test:
```
python eval_LPN_CT.py \
--pretrained_path weights/lpn_no_patch_64_bsd_noise_0.09/LPN.pt --lpn_no_patch \
--stepsize 0.015 --beta 1.0 --max_iter 100 --iterator ADMM --clamp
```
Use `accelerate launch` for multi-gpu testing.


### Experiment 3: CT reconstruction using model trained on LoDoPaB.
Train:
```
python training_LPN.py --dataset LoDoPaB --noise_level 0.1 --batch_size 64 --ckpt_dir weights/lpn_64_ct_noise_0.1
```
Weights will be saved in `weights/lpn_64_ct_noise_0.1/LPN.pt`.
Test:
```
python eval_LPN_CT.py \
--pretrained_path weights/lpn_no_patch_64_ct_noise_0.1/LPN.pt --lpn_no_patch \
--stepsize 0.02 --beta 1.0 --max_iter 100 --iterator ADMM --clamp
```


## Patch Averaging LPN

### Experiment 2: CT reconstruction using model trained on BSD.
Train:
```
python trainin_LPN.py --dataset BSD --noise_level 0.05 --batch_size 64 --ckpt_dir weights/lpn_64_bsd_noise_0.05
```
Weights will be saved in `weights/lpn_64_bsd_noise_0.05/LPN.pt`.
Test:
```
python eval_LPN_CT.py \
--pretrained_path weights/lpn_64_bsd_noise_0.05/LPN.pt \
--stepsize 0.008 --beta 1.0 --max_iter 100 --iterator ADMM --clamp --stride_size 32 --exact_prox
```

### Experiment 3: CT reconstruction using model trained on LoDoPaB.
Train:
```
python trainin_LPN.py --dataset LoDoPaB --noise_level 0.1 --batch_size 128 --ckpt_dir weights/lpn_64_ct
```
Weights will be saved in `weights/lpn_64_ct/LPN.pt`.
Test:
```
python eval_LPN_CT.py \
--pretrained_path weights/lpn_64_ct/LPN.pt \
--stepsize 0.02 --beta 1.0 --max_iter 100 --iterator ADMM --clamp --stride_size 32 --exact_prox
```
