"""
GLASS LAS (Local Anomaly Synthesis) extracted as a stand-alone callable.

Mirrors the synthesis section of `datasets.mvtec.MVTecDataset.__getitem__`
(lines 172-212) without any of the dataset-iteration / fg_mask path logic
or DataLoader wiring. The intent is that this module is the *only* place
where Perlin x DTD blending is implemented in this workspace; the GUI app
and `dump_synthetic.py` both call into here.

Output modes:
  - working_size (default 288): the resolution at which LAS runs internally.
    Must be square; Perlin mask generation is square-only upstream.
  - output_size (None or int):
        None -> return at original PIL image size (Q3 default)
        int  -> resize the final result to this square size
                (used by `dump_synthetic.py` to keep regression == 288)

The function is pure: same global RNG state in -> same output out. Caller
is expected to seed *three* RNGs before invoking:

    np.random.seed(s)
    torch.manual_seed(s)
    imgaug.seed(s)        # perlin.py uses imgaug.iaa.Affine internally

Missing the imgaug seed will give visually similar but pixel-different output
runs even with identical np/torch seeds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import PIL.Image
import torch
from torchvision import transforms

from ._vendored.perlin import perlin_mask

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class SynthParams:
    """Knobs exposed to the GUI / CLI for one synthesis call."""

    working_size: int = 288
    output_size: Optional[int] = None  # None => preserve original image size
    perlin_scale_min: int = 0  # exponent: 2**min is the smallest scale
    perlin_scale_max: int = 6  # exponent: 2**max is the largest scale
    beta_mean: float = 0.5
    beta_std: float = 0.1
    rand_aug: bool = True
    downsampling: int = 8  # working_size // downsampling -> feat_size for mask_s
    use_foreground: bool = False  # if True, fg_mask must be provided


@dataclass
class SynthResult:
    """Output of a single LAS call."""

    ng_image_bgr: np.ndarray  # H x W x 3, BGR uint8 (cv2-friendly)
    mask_uint8: np.ndarray  # H x W, 0/255 uint8 binary mask at output res
    mask_l_float: np.ndarray  # H_work x W_work, 0/1 float mask at working res
    mask_s_float: np.ndarray  # feat x feat, 0/1 float mask at feature res
    beta_used: float
    perlin_thr_kind: str  # "or", "and", or "single" (which combination was sampled)


def _build_image_transform(working_size: int) -> transforms.Compose:
    """Replicates `MVTecDataset.transform_img` for fg=0, no rotation/flip."""
    return transforms.Compose([
        transforms.Resize(working_size),
        transforms.CenterCrop(working_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _build_random_texture_transform(working_size: int) -> transforms.Compose:
    """Replicates `MVTecDataset.rand_augmenter`: 3 random aug picks from a pool."""
    list_aug = [
        transforms.ColorJitter(contrast=(0.8, 1.2)),
        transforms.ColorJitter(brightness=(0.8, 1.2)),
        transforms.ColorJitter(saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomVerticalFlip(p=1),
        transforms.RandomGrayscale(p=1),
        transforms.RandomAutocontrast(p=1),
        transforms.RandomEqualize(p=1),
        transforms.RandomAffine(degrees=(-45, 45)),
    ]
    aug_idx = np.random.choice(np.arange(len(list_aug)), 3, replace=False)
    return transforms.Compose([
        transforms.Resize(working_size),
        list_aug[aug_idx[0]],
        list_aug[aug_idx[1]],
        list_aug[aug_idx[2]],
        transforms.CenterCrop(working_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _denorm_to_bgr_uint8(img_chw: torch.Tensor) -> np.ndarray:
    """3xHxW normalized tensor -> HxWx3 BGR uint8.

    Same transformation as `utils.torch_format_2_numpy_img` for the 3-channel path.
    """
    arr = img_chw.detach().cpu().numpy()
    if arr.shape[0] != 3:
        raise ValueError(f"expected 3xHxW, got {arr.shape}")
    arr = arr.transpose(1, 2, 0)
    arr = arr * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    arr = arr[:, :, [2, 1, 0]]  # RGB -> BGR
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def _resize_image_bgr(arr: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Resize BGR uint8 image to (H, W). Avoids importing cv2 at module top."""
    import cv2
    h, w = target_hw
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)


def _resize_mask_uint8(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor upsample of a 0/255 binary mask."""
    import cv2
    h, w = target_hw
    if mask.shape[0] == h and mask.shape[1] == w:
        return mask
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def synthesize_one(
    image: PIL.Image.Image,
    texture: PIL.Image.Image,
    params: SynthParams,
    fg_mask: Optional[PIL.Image.Image] = None,
) -> SynthResult:
    """Run one LAS step. See module docstring.

    Args:
        image:    OK (normal) source image, RGB.
        texture:  Anomaly source texture, RGB.
        params:   Knobs; see SynthParams.
        fg_mask:  Optional foreground mask (any mode; will be converted to {0,1}).
                  Required iff params.use_foreground is True.
    """
    if params.working_size <= 0:
        raise ValueError("working_size must be positive")
    if params.working_size % params.downsampling != 0:
        raise ValueError(
            f"working_size ({params.working_size}) must be divisible by "
            f"downsampling ({params.downsampling})"
        )
    if params.use_foreground and fg_mask is None:
        raise ValueError("use_foreground=True but fg_mask is None")

    work = params.working_size
    feat = work // params.downsampling

    # --- 1. Transform OK image ----------------------------------------------
    img_t = _build_image_transform(work)(image.convert("RGB"))  # 3 x W x W

    # --- 2. Transform texture -----------------------------------------------
    if params.rand_aug:
        tex_t = _build_random_texture_transform(work)(texture.convert("RGB"))
    else:
        tex_t = _build_image_transform(work)(texture.convert("RGB"))

    # --- 3. Foreground mask -------------------------------------------------
    if params.use_foreground:
        # Match MVTecDataset: ToTensor then ceil to {0,1}, take channel 0.
        fg_t = transforms.Compose([
            transforms.Resize(work),
            transforms.CenterCrop(work),
            transforms.ToTensor(),
        ])(fg_mask)
        mask_fg = torch.ceil(fg_t[0])
    else:
        # Scalar tensor [1] broadcasts to ones inside perlin_mask; matches
        # MVTecDataset's `mask_fg = torch.tensor([1])` default.
        mask_fg = torch.tensor([1])

    # --- 4. Perlin mask -----------------------------------------------------
    img_shape = (3, work, work)
    mask_s_np, mask_l_np = perlin_mask(
        img_shape,
        feat,
        params.perlin_scale_min,
        params.perlin_scale_max,
        mask_fg,
        flag=1,
    )
    mask_s_t = torch.from_numpy(mask_s_np).float()
    mask_l_t = torch.from_numpy(mask_l_np).float()

    # --- 5. Beta blend ------------------------------------------------------
    beta = float(np.clip(np.random.normal(loc=params.beta_mean, scale=params.beta_std), 0.2, 0.8))
    aug_image_t = (
        img_t * (1 - mask_l_t)
        + (1 - beta) * tex_t * mask_l_t
        + beta * img_t * mask_l_t
    )

    # --- 6. To numpy uint8 BGR (working resolution) -------------------------
    ng_bgr_work = _denorm_to_bgr_uint8(aug_image_t)

    # --- 7. Resolve output resolution ---------------------------------------
    if params.output_size is None:
        target_h, target_w = image.height, image.width
    else:
        target_h = target_w = int(params.output_size)
    ng_bgr = _resize_image_bgr(ng_bgr_work, (target_h, target_w))

    # Mask: working-res mask_l_t -> uint8 -> resize to output res
    mask_l_uint8 = (mask_l_np > 0.5).astype(np.uint8) * 255
    mask_uint8 = _resize_mask_uint8(mask_l_uint8, (target_h, target_w))

    return SynthResult(
        ng_image_bgr=ng_bgr,
        mask_uint8=mask_uint8,
        mask_l_float=mask_l_np.astype(np.float32),
        mask_s_float=mask_s_np.astype(np.float32),
        beta_used=beta,
        perlin_thr_kind="combined",  # internal kind is opaque; perlin.py samples it
    )
