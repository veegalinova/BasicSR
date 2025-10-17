"""
Metrics submodule.

This module contains various metrics and a metrics tracking class.
"""

from collections.abc import Callable
from typing import Union, Optional

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

# TODO: does this add additional dependency?


# TODO revisit metric for notebook
def avg_range_invariant_psnr(
    pred: np.ndarray,
    target: np.ndarray,
) -> float:
    """Compute the average range-invariant PSNR.

    Parameters
    ----------
    pred : np.ndarray
        Predicted images.
    target : np.ndarray
        Target images.

    Returns
    -------
    float
        Average range-invariant PSNR value.
    """
    psnr_arr = []
    for i in range(pred.shape[0]):
        psnr_arr.append(scale_invariant_psnr(pred[i], target[i]))
    return np.mean(psnr_arr)


def psnr(gt: np.ndarray, pred: np.ndarray, data_range: float) -> float:
    """
    Peak Signal to Noise Ratio.

    This method calls skimage.metrics.peak_signal_noise_ratio. See:
    https://scikit-image.org/docs/dev/api/skimage.metrics.html.

    NOTE: to avoid unwanted behaviors (e.g., data_range inferred from array dtype),
    the data_range parameter is mandatory.

    Parameters
    ----------
    gt : np.ndarray
        Ground truth array.
    pred : np.ndarray
        Predicted array.
    data_range : float
        The images pixel range.

    Returns
    -------
    float
        PSNR value.
    """
    return peak_signal_noise_ratio(gt, pred, data_range=data_range)


def _zero_mean(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Zero the mean of an array.

    Parameters
    ----------
    x : numpy.ndarray or torch.Tensor
        Input array.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Zero-mean array.
    """
    return x - x.mean()


def _fix_range(
    gt: Union[np.ndarray, torch.Tensor], x: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Adjust the range of an array based on a reference ground-truth array.

    Parameters
    ----------
    gt : Union[np.ndarray, torch.Tensor]
        Ground truth array.
    x : Union[np.ndarray, torch.Tensor]
        Input array.

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Range-adjusted array.
    """
    a = (gt * x).sum() / (x * x).sum()
    return x * a


def _fix(
    gt: Union[np.ndarray, torch.Tensor], x: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Zero mean a groud truth array and adjust the range of the array.

    Parameters
    ----------
    gt : Union[np.ndarray, torch.Tensor]
        Ground truth image.
    x : Union[np.ndarray, torch.Tensor]
        Input array.

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Zero-mean and range-adjusted array.
    """
    gt_ = _zero_mean(gt)
    return _fix_range(gt_, _zero_mean(x))


def scale_invariant_psnr(
    gt: np.ndarray, pred: np.ndarray
) -> Union[float, torch.tensor]:
    """
    Scale invariant PSNR.

    Parameters
    ----------
    gt : np.ndarray
        Ground truth image.
    pred : np.ndarray
        Predicted image.

    Returns
    -------
    Union[float, torch.tensor]
        Scale invariant PSNR value.
    """
    range_parameter = (np.max(gt) - np.min(gt)) / np.std(gt)
    gt_ = _zero_mean(gt) / np.std(gt)
    return psnr(_zero_mean(gt_), _fix(gt_, pred), range_parameter)
