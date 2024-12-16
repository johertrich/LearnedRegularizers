from .datasets import BSDS500Dataset, FastMRISlices
from torchvision.transforms import ToTensor, Compose, Grayscale


def get_dataset(key, test=False, transform=None, root=None):
    if key == "BSDS500_gray":
        if transform is None:
            transforms = Compose([ToTensor(), Grayscale()])
        else:
            transforms = Compose([ToTensor(), Grayscale(), transform])
        return BSDS500Dataset(
            root="BSDS500", download=True, test=test, transform=transforms
        )
    # elif key == "BSDS500_color":
    #     if transform is None:
    #         transforms = ToTensor()
    #     else:
    #         transforms = Compose([ToTensor(), transform])
    #     return BSDS500Dataset(
    #         root="BSDS500", download=True, test=test, transform=transforms
    #     )
    elif key == "fastMRI":
        if transform is None:
            transforms = ToTensor()
        else:
            transforms = Compose([ToTensor(), transform])
        return FastMRISlices(root, test=test, transform=transforms)
    else:
        raise NameError("Unknown dataset!")
