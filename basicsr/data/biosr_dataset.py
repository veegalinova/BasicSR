from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
from basicsr.data.transforms import augment

from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class BioSRDataset(Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(BioSRDataset, self).__init__()
        self.opt = opt
        self.input_mean = opt["input_mean"]
        self.input_std = opt["input_std"]
        self.target_mean = opt["target_mean"]
        self.target_std = opt["target_std"]

        self.folder = Path(opt["dataroot"])
        self.paths = sorted(self.folder.glob('*.tif*'))

        self.normalize_input = Normalize(self.input_mean, self.input_std)
        self.normalize_target = Normalize(self.target_mean, self.target_std)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = tifffile.imread(path)
        img_gt, img_lq = img

        # augmentation for training
        if self.opt["phase"] == "train":
            # flip, rotation
            img_gt, img_lq = augment(
                [img_gt, img_lq], self.opt["use_hflip"], self.opt["use_rot"]
            )

        img_gt = torch.from_numpy(img_gt).to(torch.float32)
        img_lq = torch.from_numpy(img_lq).to(torch.float32)

        img_gt = img_gt.unsqueeze(0)
        img_lq = img_lq.unsqueeze(0)

        # # normalize
        if self.input_mean is not None or self.input_std is not None:
            img_lq = self.normalize_input(img_lq)
            img_gt = self.normalize_target(img_gt)

        return {"lq": img_lq, "gt": img_gt, "path": str(path)}
