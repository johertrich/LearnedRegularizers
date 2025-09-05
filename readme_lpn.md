## Experiment 1: Denoising on BSD
Train:
```
python trainin_LPN.py --dataset BSD --noise_level 0.1 --batch_size 64 --ckpt_dir weights/lpn_64_bsd
```
Weights will be saved in `weights/lpn_64_bsd/LPN.pt`. (`lpn_64` stands for patch size 64.)
Test:
```
python eval_LPN.py --task denoising
```

## Experiment 2: CT reconstruction using model trained on BSD.
Train:
```
python trainin_LPN.py --dataset BSD --noise_level 0.05 --batch_size 64 --ckpt_dir weights/lpn_64_bsd_noise_0.05
```
Weights will be saved in `weights/lpn_64_bsd_noise_0.05/LPN.pt`.
Test:
```
python eval_LPN.py --task ct_trained_on_bsd
```

## Experiment 3: CT reconstruction using model trained on LoDoPaB.
Train:
```
python trainin_LPN.py --dataset LoDoPaB --noise_level 0.1 --batch_size 128 --ckpt_dir weights/lpn_64_ct
```
Weights will be saved in `weights/lpn_64_ct/LPN.pt`.
Test:
```
python eval_LPN.py --task ct
```