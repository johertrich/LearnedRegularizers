import urllib.request
from torch.utils.data import Dataset
import zipfile
import os
import os.path
from PIL import Image
import numpy as np
import h5py


class BSDS500Dataset(Dataset):
    def __init__(self, root, download=True, test=False, transform=None, rotate=False):
        self.base_path = root
        self.rotate = rotate
        self.transforms = transform
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        zip_path = os.path.join(self.base_path, "download.zip")
        if download and not os.path.exists(zip_path):
            urllib.request.urlretrieve(
                "https://github.com/BIDS/BSDS500/archive/refs/heads/master.zip",
                zip_path,
            )
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.base_path)
        if not download and not os.path.exists(zip_path):
            raise NameError(
                "Dataset does not exist. Set download=True for downloading it."
            )
        image_path_train = os.path.join(
            self.base_path, "BSDS500-master/BSDS500/data/images/train"
        )
        image_path_val = os.path.join(
            self.base_path, "BSDS500-master/BSDS500/data/images/test"
        )
        image_path_test = os.path.join(
            self.base_path, "BSDS500-master/BSDS500/data/images/val"
        )
        if test:
            file_list = os.listdir(image_path_test)
            self.file_list = [
                os.path.join(image_path_test, f) for f in file_list if f.endswith("jpg")
            ]
        else:
            file_list = os.listdir(image_path_train)
            self.file_list = [
                os.path.join(image_path_train, f)
                for f in file_list
                if f.endswith("jpg")
            ]
            file_list = os.listdir(image_path_val)
            self.file_list = self.file_list + [
                os.path.join(image_path_val, f) for f in file_list if f.endswith("jpg")
            ]
        self.file_list.sort()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, IDX):
        img = Image.open(self.file_list[IDX]).convert("RGB")
        img = np.array(img) / 255.0
        if self.transforms is not None:
            img = self.transforms(img)
        if self.rotate:
            if isinstance(img, (tuple, list)):
                img = [
                    i.transpose(-2, -1) if i.shape[-1] > i.shape[-2] else i for i in img
                ]
            else:
                img = img.transpose(-2, -1) if img.shape[-1] > img.shape[-2] else img
        return img


class BSD68(Dataset):
    def __init__(self, transform=None):
        self.transforms = transform
        image_path = "dataset/BSD68"
        file_list = os.listdir(image_path)
        self.file_list = [
            os.path.join(image_path, f) for f in file_list if f.endswith("jpg")
        ]
        self.file_list.sort()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, IDX):
        img = Image.open(self.file_list[IDX]).convert("RGB")
        img = np.array(img) / 255.0
        if self.transforms is not None:
            img = self.transforms(img)
        return img


class LoDoPaB(Dataset):
    def __init__(self, root, download=True, test=False, transform=None):
        self.base_path = root
        self.transforms = transform
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        zip_path1 = os.path.join(self.base_path, "download1.zip")
        zip_path2 = os.path.join(self.base_path, "download2.zip")
        if download and not os.path.exists(zip_path1):
            print(
                "Download dataset part 1 of 2. Part 1 has total size of about 1.6 GB, hence the download might take some time..."
            )
            urllib.request.urlretrieve(
                "https://zenodo.org/records/3384092/files/ground_truth_test.zip",
                zip_path1,
            )
            print("Download of part 1 completed...")
            print(
                "Download dataset part 2 of 2. Part 2 has total size of about 1.6 GB, hence the download might take some time..."
            )
            urllib.request.urlretrieve(
                "https://zenodo.org/records/3384092/files/ground_truth_validation.zip",
                zip_path2,
            )
            print("Download of part 2 completed, extracting dataset...")
            with zipfile.ZipFile(zip_path1, "r") as zip_ref:
                zip_ref.extractall(self.base_path)
            with zipfile.ZipFile(zip_path2, "r") as zip_ref:
                zip_ref.extractall(self.base_path)
            print("Dataset extracted.")
        if not download and (
            not os.path.exists(zip_path1) or not os.path.exists(zip_path2)
        ):
            raise NameError(
                "Dataset does not exist. Set download=True for downloading it."
            )
        self.test = test
        if test:
            self.length = 128
        else:
            fname = "ground_truth_validation_027.hdf5"
            with h5py.File(os.path.join(self.base_path, fname), "r") as f:
                batch = f["data"][()]
            self.length = 128 * 27 + np.sum(np.sum(np.abs(batch), (1, 2)) > 0)

    def __len__(self):
        return self.length

    def __getitem__(self, IDX):
        batch_idx = IDX // 128
        img_idx = IDX % 128
        if self.test:
            fname = "ground_truth_test_000.hdf5"
        else:
            idx_str = f"{batch_idx}"
            while len(idx_str) < 3:
                idx_str = "0" + idx_str
            fname = "ground_truth_validation_" + idx_str + ".hdf5"
        with h5py.File(os.path.join(self.base_path, fname), "r") as f:
            img = f["data"][img_idx, :, :]
        if self.transforms is not None:
            img = self.transforms(img)
        return img