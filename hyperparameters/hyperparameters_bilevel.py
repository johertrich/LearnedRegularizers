import argparse


def get_bilevel_hyperparameters(regularizer_name, problem):
    if not problem in ["Denoising", "CT"]:
        raise ValueError("Unknown Problem!")
    if not regularizer_name in ["CRR", "WCRR", "ICNN", "IDCNN", "LAR", "TDV", "LSR"]:
        raise ValueError("Unknown Regularizer!")
    args = argparse.Namespace()

    # hyperparmaters which are the same for all settings:
    args.score_sigma = (
        3e-2 if problem == "Denoising" else 1.5e-2
    )  # noise level for pretraining
    args.adabelief = True  # chooses optimizer
    args.jacobian_regularization = True  # whether to use Jacobian regularization

    if regularizer_name in ["CRR", "WCRR"]:

        args.pretrain_weight_decay = 0  # weight decay in pretraining
        args.pretrain_lr = 1e-2  # learning rate in pretraining
        args.fitting_lr = 0.1  # learning rate in the parameter fitting phase
        args.lr = 1e-3  # learning rate in the bilevel phase
        args.do_parameter_fitting = True

        if problem == "Denoising":
            args.pretrain_epochs = 300  # number of epochs in pretraining
            args.jacobian_regularization_parameter = (
                1e-6  # Jacobian regularization parameter
            )
            args.epochs = 100  # number of epochs in the bilevel phase

        if problem == "CT":
            args.pretrain_epochs = 40  # number of epochs in pretraining
            args.jacobian_regularization_parameter = (
                1e-8  # Jacobian regularization parameter
            )
            args.epochs = 4  # number of epochs in the bilevel phase

    if regularizer_name == "ICNN":

        args.pretrain_weight_decay = 0  # weight decay in pretraining
        args.pretrain_lr = 1e-3  # learning rate in pretraining
        args.fitting_lr = 0.1  # learning rate in the parameter fitting phase
        args.lr = 1e-3  # learning rate in the bilevel phase
        args.do_parameter_fitting = True

        if problem == "Denoising":
            args.pretrain_epochs = 300  # number of epochs in pretraining
            args.jacobian_regularization_parameter = (
                1e-6  # Jacobian regularization parameter
            )
            args.epochs = 200  # number of epochs in the bilevel phase

        if problem == "CT":
            args.pretrain_epochs = 40  # number of epochs in pretraining
            args.jacobian_regularization_parameter = (
                1e-7  # Jacobian regularization parameter
            )
            args.epochs = 4  # number of epochs in the bilevel phase

    if regularizer_name == "IDCNN":

        args.pretrain_weight_decay = 0  # weight decay in pretraining
        args.pretrain_lr = 1e-3  # learning rate in pretraining
        args.do_parameter_fitting = True

        if problem == "Denoising":
            args.pretrain_epochs = 300  # number of epochs in pretraining
            args.jacobian_regularization_parameter = (
                1e-5  # Jacobian regularization parameter
            )
            args.epochs = 200  # number of epochs in the bilevel phase
            args.fitting_lr = 0.01  # learning rate in the parameter fitting phase
            args.lr = 1e-3  # learning rate in the bilevel phase

        if problem == "CT":
            args.pretrain_epochs = 40  # number of epochs in pretraining
            args.jacobian_regularization_parameter = (
                1e-8  # Jacobian regularization parameter
            )
            args.epochs = 10  # number of epochs in the bilevel phase
            args.fitting_lr = 0.1  # learning rate in the parameter fitting phase
            args.lr = 1e-4  # learning rate in the bilevel phase

    if regularizer_name == "LAR":

        if problem == "Denoising":
            args.pretrain_weight_decay = 0  # weight decay in pretraining
            args.pretrain_lr = 1e-3  # learning rate in pretraining
            args.fitting_lr = 0.01  # learning rate in the parameter fitting phase
            args.lr = 1e-3  # learning rate in the bilevel phase
            args.pretrain_epochs = 300  # number of epochs in pretraining
            args.jacobian_regularization_parameter = (
                1e-5  # Jacobian regularization parameter
            )
            args.epochs = 200  # number of epochs in the bilevel phase
            args.do_parameter_fitting = True

        if problem == "CT":
            raise ValueError("We did not run the LAR for CT...")

    if regularizer_name == "TDV":

        args.pretrain_weight_decay = 0  # weight decay in pretraining
        args.pretrain_lr = 4e-4  # learning rate in pretraining

        if problem == "Denoising":
            args.pretrain_epochs = 7500  # number of epochs in pretraining
            args.jacobian_regularization_parameter = (
                1e-4  # Jacobian regularization parameter
            )
            args.epochs = 200  # number of epochs in the bilevel phase
            args.fitting_lr = 0.005  # learning rate in the parameter fitting phase
            args.lr = 1e-4  # learning rate in the bilevel phase
            args.do_parameter_fitting = True

        if problem == "CT":
            args.pretrain_epochs = 750  # number of epochs in pretraining
            args.jacobian_regularization_parameter = (
                1e-8  # Jacobian regularization parameter
            )
            args.epochs = 10  # number of epochs in the bilevel phase
            args.fitting_lr = 0.05  # learning rate in the parameter fitting phase
            args.lr = 5e-5  # learning rate in the bilevel phase
            args.do_parameter_fitting = False

    if regularizer_name == "LSR":
        args.pretrain_lr = 2e-4  # learning rate in pretraining

        if problem == "Denoising":
            args.pretrain_epochs = 7500  # number of epochs in pretraining
            args.jacobian_regularization_parameter = (
                1e-4  # Jacobian regularization parameter
            )
            args.epochs = 200  # number of epochs in the bilevel phase
            args.fitting_lr = 0.05  # learning rate in the parameter fitting phase
            args.lr = 1e-4  # learning rate in the bilevel phase
            args.pretrain_weight_decay = 0  # weight decay in pretraining
            args.do_parameter_fitting = True

        if problem == "CT":
            args.pretrain_epochs = 750  # number of epochs in pretraining
            args.jacobian_regularization_parameter = (
                1e-8  # Jacobian regularization parameter
            )
            args.epochs = 10  # number of epochs in the bilevel phase
            args.fitting_lr = 0.05  # learning rate in the parameter fitting phase
            args.lr = 5e-6  # learning rate in the bilevel phase
            args.pretrain_weight_decay = 1e-4  # weight decay in pretraining
            args.do_parameter_fitting = True

    return args
