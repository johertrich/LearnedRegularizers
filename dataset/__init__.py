from .datasets import BSDS500Dataset, BSD68, LoDoPaB
from torchvision.transforms.v2 import ToImage, Compose, Grayscale, ToDtype
import torch
import os


def get_dataset(key, test=False, transform=None, root=None, rotate=False):
    """
    Load the datasets used in the chapter as torch.data.Dataset object.

    Available datasets are:

    - BSD68: A gray-valued version of BSD68
    - BSDS500_gray: gray-valued Berkley Segmentation Dataset (BSDS500)
    - LoDoPaB: A subset of the LoDoPaB CT dataset. The training split corresponds to the original validation split and the test split is the first batch of the original test split
    - LoDoPaB_val: five images from the train split of LoDoPaB which are used to fit the parameters in Experiment 2

    Note that this routine automatically downloads the dataset if it is not available at the specified root directory.

    Input arguments:
    key - specify which dataset shoud be loaded. Choices are "BSDS500_gray", "BSD68", "LoDoPaB" and "LoDoPaB_val"
    test - if True, we use the test split, if False we use the Train split for the dataset
    transform - any transforms (e.g. via torchvision)
    root - optionally define the base directory, where the dataset is located (or the destination where it will
        be downloaded). If None, "." will be used
    rotate - only relevant for the BSD68 and BSDS500 dataset. If set to True images are rotated such that all images
        in the dataset have the same orientation (portrait or landscape).
    """
    if root is not None:
        location = os.path.join(root, key)
    else:
        location = key
    if key == "BSDS500_gray":
        location = location[:-5]
        if transform is None:
            transforms = Compose(
                [ToImage(), ToDtype(torch.float32, scale=True), Grayscale()]
            )
        else:
            transforms = Compose(
                [ToImage(), ToDtype(torch.float32, scale=True), Grayscale(), transform]
            )
        return BSDS500Dataset(
            root=location, download=True, test=test, transform=transforms, rotate=rotate
        )
    elif key == "BSD68":
        if transform is None:
            transforms = Compose(
                [ToImage(), ToDtype(torch.float32, scale=True), Grayscale()]
            )
        else:
            transforms = Compose(
                [ToImage(), ToDtype(torch.float32, scale=True), Grayscale(), transform]
            )
        return BSD68(transform=transforms)
    if key == "LoDoPaB":
        if transform is None:
            transforms = Compose([ToImage(), ToDtype(torch.float32, scale=True)])
        else:
            transforms = Compose(
                [ToImage(), ToDtype(torch.float32, scale=True), transform]
            )
        return LoDoPaB(root=location, download=True, test=test, transform=transforms)
    if key == "LoDoPaB_val":
        if root is not None:
            location = os.path.join(root, "LoDoPaB")
        else:
            location = "LoDoPaB"
        if transform is None:
            transforms = Compose([ToImage(), ToDtype(torch.float32, scale=True)])
        else:
            transforms = Compose(
                [ToImage(), ToDtype(torch.float32, scale=True), transform]
            )
        ds = LoDoPaB(root=location, download=True, test=False, transform=transforms)
        return torch.utils.data.Subset(ds, range(5))
    else:
        raise NameError("Unknown dataset!")
