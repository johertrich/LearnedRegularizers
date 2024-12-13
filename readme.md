# Learned Variational Regularization: A Comparative Study 

The code relies on the [DeepInverse package](https://deepinv.github.io), which can be installed as follows:

```
pip install deepinv
```

## What is currently in there

There is

- a script `examples.py` which evaluates a denoising or a MRI (toy version with real-valued images) task on the BSDS500 dataset with the unrolled ICNN
- a script `training_simple_ICNN_unrolling.py` training a small ICNN via unrolling
- a `dataset.get_dataset(key, test=False, transform=None)` method to select a dataset. The main idea of using a centralized dataset generation is to make sure that everyone really uses exactly the same datasets. Currently the available keys are `"BSDS500_gray"` and `"BSDS500_color"`.

Helper functions for the evaluation script:

- in operators there is the MRI operator `operators.MRIonR` (it wraps the `deepinv` version, which considers complex-valued images; but I think, we should stay with real-valued images, even though its a bit academic).
- a `evaluation.evaluate` implements the evaluation procedure with some Nesterov accelerated gradient algorithm and iterates the evaluation over the test dataset.

## How to add something to the repo

In order to coordinate the implementations please respect the following structure of the repo:

- Architectures should be implemented in the directory `priors`, training methods in `training_methods`. Please create in `priors` and `training_methods` exactly one file for every architecture/training method. If you require additional files, you can create subdirectories and put the additional there.
- Example scripts for training the methods should be created top-level.
- To avoid confusions, it would be best if everyone only edits the files, which they have created/written. If you would like to edit existing files, please ask the person who created/has written them.
- generally all images have the shape `(B,C,H,W)` with batch dimension `B`, channel dimension `C`, height `H` and width `W`.

If anything is unclear, you have questions, or you need help, contact me (Johannes). If you prefer you can also make any edits in a different branch and create a pull request.

### Structure of Architectures

Architectures should inherit from [`deepinv.optim.Prior`](https://deepinv.github.io/deepinv/api/stubs/deepinv.optim.Prior.html) (which inherits from `torch.nn.Module`). They should have:

- An `__init__` method, which takes all hyperparameters and a keyword argument `pretrained`. If `pretrained` is a string the init method should load weights from the path which is defined by the `pretrained` argument.
- A function `g(self, x)` which evaluates the regularizer.
- A function `grad(self, x)` which evaluates the gradient of the regularizer
- Do we need the following for the bilevel methods?: A function `hvp(self,x)` which evaluates the Hessian vector product.

### Structure of Training Methods

Training methods should be callable functions. To ensure the interoperability for different architectures they should have the positional arguments:

- `regularizer`: The regularizer which is trained
- `physics`: Defining the forward operator of the variational problem. This is a [`deepinv.physics.LinearPhysics`](https://deepinv.github.io/deepinv/api/stubs/deepinv.physics.LinearPhysics.html) object. That is, it implements
    + the forward operator as `physics.A` and its adjoint as `physics.A_adjoint`
    + the `physics(x)` applies the forward operator and the noise model to `x`.
- `data_fidelitiy`: defining the data-fidelity term of the variational problem. This is a [`deepinv.optim.DataFidelity`](https://deepinv.github.io/deepinv/api/stubs/deepinv.optim.DataFidelity.html) implementing
    + `data_fidelity(x,y,physics)`: evaluates $d(Ax,y)$ where $A$ is the forward operator
    + `data_fidelity.grad(x,y,physics)`: evaluates the gradient of $d(Ax,y)$ wrt x (backprobagation through `data_fidelity(x,y,physics)` should work as well)
- `lmbd`: regularization parameter of the variational problem
- `train_dataloader`: A dataloader for the training dataset
- `val_dataloader`: A dataloader for the validation set

Any other hyperparameters should be keyword arguments, where the defaults are adjusted to the denoising problem on BSD500 with noise level 0.1.

