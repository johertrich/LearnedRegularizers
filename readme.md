# Learned Variational Regularization: A Comparative Study 

The code relies on the [DeepInverse package](https://deepinv.github.io), which can be installed as follows:

```
pip install deepinv
```

## What is currently in there

There is

- a script `examples.py` which evaluates a denoising or a MRI (toy version with real-valued images) task on the BSDS500 dataset
- a `dataset.get_dataset(key, test=False, transform=None)` method to select a dataset. The main idea of using a centralized dataset generation is to make sure that everyone really uses exactly the same datasets. Currently the available keys are `"BSDS500_gray"` and `"BSDS500_color"`.

Helper functions for the evaluation script:

- in operators there is the MRI operator `operators.MRIonR` (it wraps the `deepinv` version, which considers complex-valued images; but I think, we should stay with real-valued images, even though its a bit academic).
- a `evaluation.evaluate` implements the evaluation procedure with some Nesterov accelerated gradient algorithm and iterates the evaluation over the test dataset.