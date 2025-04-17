import urllib.request
from torch.utils.data import Dataset
import zipfile
import os
import os.path
from PIL import Image
import numpy as np
import h5py


class BSDS500Dataset(Dataset):
    def __init__(self, root, download=True, test=False, transform=None):
        self.base_path = root
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
            self.base_path, "BSDS500-master/BSDS500/data/images/val"
        )
        image_path_test = os.path.join(
            self.base_path, "BSDS500-master/BSDS500/data/images/test"
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


class FastMRISlices(Dataset):
    def __init__(self, root, test=False, transform=None):
        if root is None:
            if test:
                root = "fastMRI/knee_singlecoil_val/singlecoil_val"
            else:
                root = "fastMRI/knee_singlecoil_train/singlecoil_train"
        self.root = root
        self.transforms = transform
        file_list = os.listdir(root)
        self.file_list = [f for f in file_list if f[-3:] == ".h5"]
        self.file_list.sort()
        self.num_slices = []
        self.slice_inds = []
        # cut out the first/last 5 slices, since they are often not good training images
        cut = 5
        for fname in self.file_list:
            with h5py.File(os.path.join(self.root, fname), "r") as f:
                n_slices = f["reconstruction_esc"].shape[0]
                if test:  # take only the middle slice for testing
                    self.num_slices.append(1)
                    self.slice_inds.append([n_slices // 2])
                else:
                    self.num_slices.append(n_slices - 2 * cut)
                    self.slice_inds.append(list(range(cut, n_slices - cut)))
        self.num_slices = np.array(self.num_slices)
        self.num_slices_cumsum = np.cumsum(self.num_slices)

    def __len__(self):
        return self.num_slices_cumsum[-1]

    def __getitem__(self, IDX):
        fnum = np.argmax(self.num_slices_cumsum > IDX)
        slice_ind = IDX - self.num_slices_cumsum[fnum] + self.num_slices[fnum]
        slice_num = self.slice_inds[fnum][slice_ind]
        fname = self.file_list[fnum]
        with h5py.File(os.path.join(self.root, fname), "r") as f:
            img = f["reconstruction_esc"][slice_num, :, :]
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
        if not download and not os.path.exists(zip_path):
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
        batch_idx = IDX // 128 + 1
        img_idx = IDX % 128
        if self.test:
            fname = "ground_truth_test_000.hdf5"
        else:
            idx_str = str(batch_idx)
            while len(idx_str) < 3:
                idx_str = "0" + idx_str
            fname = "ground_truth_validation_" + idx_str + ".hdf5"
        with h5py.File(os.path.join(self.base_path, fname), "r") as f:
            img = f["data"][img_idx, :, :]
        if self.transforms is not None:
            img = self.transforms(img)
        return img
