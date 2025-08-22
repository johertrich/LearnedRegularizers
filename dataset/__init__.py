from .datasets import BSDS500Dataset, BSD68, LoDoPaB
from torchvision.transforms.v2 import ToImage, Compose, Grayscale, ToDtype
import torch
import os


def get_dataset(key, test=False, transform=None, root=None, rotate=False):
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
    # elif key == "BSDS500_color":
    #     if transform is None:
    #         transforms = Compose([ToImage(), ToDtype(torch.float32, scale=True)])
    #     else:
    #         transforms = Compose([ToImage(), ToDtype(torch.float32, scale=True), transform])
    #     return BSDS500Dataset(
    #         root="BSDS500", download=True, test=test, transform=transforms
    #     )
    else:
        raise NameError("Unknown dataset!")