import argparse


def get_AR_hyperparameters(regularizer_name, problem):
    if not problem in ["Denoising", "CT"]:
        raise ValueError("Unknown Problem!")
    if not regularizer_name in ["CRR", "WCRR", "ICNN", "IDCNN", "TDV", "LSR", "LAR"]:
        raise ValueError("Unknown Regularizer!")
    args = argparse.Namespace()

    if regularizer_name in ["CRR", "WCRR"]:
        args.patch_size = 64
        args.patch_per_img = 8
        args.batch_size = 8
        args.lr = 1e-2
        args.mu = 10
        args.fitting_lr = 0.1
        if problem == "Denoising":
            args.lr_decay = 0.998
            args.epochs = 500
            args.val_epochs = 25
        if problem == "CT":
            args.lr_decay = 0.98
            args.epochs = 150
            args.val_epochs = 10

    if regularizer_name == "ICNN":
        args.patch_size = 64
        args.patch_per_img = 8
        args.batch_size = 8
        args.lr = 1e-3
        args.mu = 10
        args.fitting_lr = 0.1
        if problem == "Denoising":
            args.lr_decay = 0.998
            args.epochs = 500
            args.val_epochs = 25
        if problem == "CT":
            args.lr_decay = 0.985
            args.epochs = 200
            args.val_epochs = 10

    if regularizer_name == "IDCNN":
        args.lr = 1e-3
        args.mu = 10
        args.fitting_lr = 1e-2
        if problem == "Denoising":
            args.patch_size = 64
            args.patch_per_img = 8
            args.batch_size = 32
            args.lr_decay = 0.998
            args.epochs = 1000
            args.val_epochs = 5
        if problem == "CT":
            args.patch_size = 76
            args.patch_per_img = 4
            args.batch_size = 32
            args.lr_decay = 1.0
            args.epochs = 200
            args.val_epochs = 10

    if regularizer_name == "TDV":
        args.lr = 2e-4
        args.mu = 10
        args.fitting_lr = 1e-2
        if problem == "Denoising":
            args.patch_size = 64
            args.patch_per_img = 8
            args.batch_size = 8
            args.lr_decay = 0.9997
            args.epochs = 4000
            args.val_epochs = 100
        if problem == "CT":
            args.patch_size = 25
            args.patch_per_img = 32
            args.batch_size = 8
            args.lr_decay = 0.995
            args.epochs = 500
            args.val_epochs = 15

    if regularizer_name == "LSR":
        args.lr = 1e-4
        args.fitting_lr = 1e-2
        if problem == "Denoising":
            args.patch_size = 64
            args.patch_per_img = 8
            args.batch_size = 16
            args.lr_decay = 0.9997
            args.epochs = 4000
            args.val_epochs = 100
            args.mu = 15
        if problem == "CT":
            args.patch_size = 25
            args.patch_per_img = 32
            args.batch_size = 8
            args.lr_decay = 0.995
            args.epochs = 500
            args.val_epochs = 15
            args.mu = 10

    if regularizer_name == "LAR":
        args.patch_size = 15
        args.patch_per_img = 64
        args.batch_size = 8
        args.val_epochs = 25
        if problem == "CT":
            args.epochs = 400
            args.fitting_lr = 1e-3
            args.mu = 0.5
            args.lr = 5e-4
        if problem == "Denoising":
            args.epochs = 1000
            args.fitting_lr = 1e-2
            args.mu = 0.1
            args.lr = 1e-3
        args.lr_decay = 0.1 ** (1 / args.epochs)

    return args
