from .datasets import BSDS500Dataset, FastMRISlices, BSD68
from torchvision.transforms.v2 import ToTensor, Compose, Grayscale, ToDtype
import torch


def get_dataset(key, test=False, transform=None, root=None):
    if key == "BSDS500_gray":
        if transform is None:
            transforms = Compose([ToTensor(), ToDtype(torch.float32), Grayscale()])
        else:
            transforms = Compose([ToTensor(), ToDtype(torch.float32), Grayscale(), transform])
        return BSDS500Dataset(
            root="BSDS500", download=True, test=test, transform=transforms
        )
    elif key == "BSD68":
        if transform is None:
            transforms = Compose([ToTensor(), ToDtype(torch.float32), Grayscale()])
        else:
            transforms = Compose([ToTensor(), ToDtype(torch.float32), Grayscale(), transform])
        return BSD68(transform=transforms)
    # elif key == "BSDS500_color":
    #     if transform is None:
    #         transforms = Compose([ToTensor(), ToDtype(torch.float32)])
    #     else:
    #         transforms = Compose([ToTensor(), ToDtype(torch.float32), transform])
    #     return BSDS500Dataset(
    #         root="BSDS500", download=True, test=test, transform=transforms
    #     )
    # elif key == "fastMRI":
    #     if transform is None:
    #         transforms = Compose([ToTensor(), ToDtype(torch.float32)])
    #     else:
    #         transforms = Compose([ToTensor(), ToDtype(torch.float32), ToDtype(), transform])
    #     return FastMRISlices(root, test=test, transform=transforms)
    else:
        raise NameError("Unknown dataset!")
