"""PyTorch dataset, augmentation pipelines and dataloaders for off-road segmentation.

Design choices come from the exploratory data analysis:
- horizontal flips are valid (left-right symmetric scenes), vertical flips are not
  (the sky is always at the top);
- illumination varies a lot, so photometric augmentations are applied to the image only;
- colour identifies classes (e.g. green vegetation), so hue is never shifted;
- the horizon is always horizontal, so rotations are limited to a few degrees.
"""
from __future__ import annotations
from pathlib import Path
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from . import config
from src.data import colorize, load_sample






def resize_pair(image: np.ndarray, mask: np.ndarray, image_size: config.ImageSize):
    """Resize an image/mask pair to ``image_size`` = (height, width).

    The image uses area interpolation (best when shrinking). The mask uses
    nearest neighbour so that it never contains invented class indices.
    """
    height, width = image_size
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return image, mask


def compute_normalization(samples: list[Path], image_size: config.ImageSize | None = config.DEFAULT_IMAGE_SIZE):
    """Per-channel mean and std (in [0, 1]) of the given samples.

    Call it with the TRAINING samples only: using the validation images would leak
    information about them into the preprocessing.

    Returns:
        (mean, std), each a tuple of three floats in RGB order.
    """
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    n_pixels = 0
    for sample in tqdm(samples, desc="Computing mean/std"):
        image, mask = load_sample(sample)
        if image_size is not None:
            image, _ = resize_pair(image, mask, image_size)
        pixels = image.reshape(-1, 3).astype(np.float64) / 255.0
        channel_sum += pixels.sum(0)
        channel_sq_sum += (pixels ** 2).sum(0)
        n_pixels += pixels.shape[0]
    mean = channel_sum / n_pixels
    std = np.sqrt(channel_sq_sum / n_pixels - mean ** 2)
    return tuple(mean.tolist()), tuple(std.tolist())


def _augmentation_steps(profile: dict) -> list:
    """Random transforms of one augmentation profile of ``config.AUGMENTATIONS``.

    The five basic transforms are always present. Blur, noise and dropout are added only when their
    probability is above zero: this keeps the "standard" pipeline (and so its random stream)
    identical to the one used by experiments A, B, C and E.
    """
    steps = [
        A.HorizontalFlip(p=profile["flip_p"]),
        A.Affine(scale=profile["scale"], rotate=profile["rotate"], translate_percent=profile["translate"],
                 fill=0, fill_mask=0, p=profile["affine_p"]),
        A.RandomBrightnessContrast(brightness_limit=profile["brightness_limit"],
                                   contrast_limit=profile["contrast_limit"], p=profile["brightness_contrast_p"]),
        A.RandomGamma(gamma_limit=profile["gamma_limit"], p=profile["gamma_p"]),
        A.ColorJitter(brightness=0.0, contrast=0.0, saturation=profile["saturation"], hue=0.0,
                      p=profile["saturation_p"]),
    ]
    if profile["blur_p"] > 0:
        steps.append(A.GaussianBlur(sigma_limit=profile["blur_sigma"], p=profile["blur_p"]))
    if profile["noise_p"] > 0:
        steps.append(A.GaussNoise(std_range=profile["noise_std"], p=profile["noise_p"]))
    if profile["dropout_p"] > 0:
        # Holes are black in the image and the mask is not touched (fill_mask=None): the network must
        # still predict the right label under the hole, using the context around it
        steps.append(A.CoarseDropout(num_holes_range=profile["dropout_holes"],
                                     hole_height_range=profile["dropout_size"],
                                     hole_width_range=profile["dropout_size"],
                                     fill=0, fill_mask=None, p=profile["dropout_p"]))
    return steps


def build_transforms(mean, std, train: bool, seed: int | None = None,
                     augmentation: str = config.AUGMENTATION) -> A.Compose:
    """Build the augmentation + normalisation pipeline.

    Geometric transforms act on image and mask together (nearest interpolation on
    the mask, padded areas labelled 0 = background, which the metric ignores).
    Photometric transforms act on the image only.

    Args:
        train: if False only normalisation and tensor conversion are applied.
        seed: seed of the random generator of the pipeline (None = random).
        augmentation: name of a profile of ``config.AUGMENTATIONS`` (used only when ``train`` is True).
    """
    steps = []
    if train:
        if augmentation not in config.AUGMENTATIONS:
            raise ValueError(f"unknown augmentation '{augmentation}', available: {sorted(config.AUGMENTATIONS)}")
        steps += _augmentation_steps(config.AUGMENTATIONS[augmentation])
    steps += [A.Normalize(mean=mean, std=std, max_pixel_value=255.0), ToTensorV2()]
    return A.Compose(steps, seed=seed)


class SegmentationDataset(Dataset):
    """Semantic segmentation dataset.

    Each item is a tuple ``(image, mask)``:
        image: float32 tensor (3, H, W), normalised.
        mask:  int64 tensor (H, W) with class indices 0..8.

    Args:
        samples: sample directories, each containing rgb.jpg and labels.png.
        transform: albumentations pipeline applied to image and mask.
        image_size: (height, width) to resize to when loading, or None to keep the
            original resolution.
        cache: keep all (resized) samples in RAM to avoid decoding the files every epoch.
    """

    def __init__(self, samples: list[Path], transform: A.Compose,
                image_size: config.ImageSize | None = config.DEFAULT_IMAGE_SIZE, cache: bool = True):
        self.samples = list(samples)
        self.transform = transform
        self.image_size = image_size
        self._cache = None
        if cache:
            self._cache = [self._read(s) for s in tqdm(self.samples, desc="Caching samples")]

    def _read(self, sample: Path):
        image, mask = load_sample(sample)
        if self.image_size is not None:
            image, mask = resize_pair(image, mask, self.image_size)
        return image, mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        if self._cache is not None:
            image, mask = self._cache[index]
        else:
            image, mask = self._read(self.samples[index])
        out = self.transform(image=image, mask=mask)
        return out["image"], out["mask"].long()


def _seed_worker(worker_id: int) -> None:
    """Give every DataLoader worker its own, reproducible augmentation seed."""
    info = torch.utils.data.get_worker_info()
    info.dataset.transform.set_random_seed(info.seed % (2 ** 32))


def create_dataloaders(train_samples: list[Path], val_samples: list[Path],
                        image_size: config.ImageSize | None = config.DEFAULT_IMAGE_SIZE, batch_size: int = 8,
                        mean=None, std=None, num_workers: int = 0, cache: bool = True,
                        seed: int = 42, augmentation: str = config.AUGMENTATION):
    """Create the training and validation dataloaders.

    If ``mean``/``std`` are not given they are computed on the training samples.
    ``augmentation`` is the name of the training augmentation profile (see ``config.AUGMENTATIONS``),
    the validation images are never augmented.

    Returns:
        (train_loader, val_loader, mean, std). Keep mean and std: the same values must
        be used at inference time.
    """
    print(f"Creating dataloaders with image size {image_size}, batch size {batch_size}, "
        f"num_workers {num_workers}, cache={cache}, seed={seed}, augmentation={augmentation}")
    if mean is None or std is None:
        mean, std = compute_normalization(train_samples, image_size)

    train_set = SegmentationDataset(train_samples, build_transforms(mean, std, True, seed, augmentation),
                                    image_size, cache)
    val_set = SegmentationDataset(val_samples, build_transforms(mean, std, False), image_size, cache)

    generator = torch.Generator()
    generator.manual_seed(seed)
    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True,  generator=generator, worker_init_fn=_seed_worker, **common)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, mean, std


def denormalize(image: torch.Tensor, mean, std) -> np.ndarray:
    """Convert a normalised (3, H, W) tensor back to an (H, W, 3) uint8 image."""
    mean_t = torch.tensor(mean, dtype=image.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=image.dtype).view(3, 1, 1)
    image = (image * std_t + mean_t).clamp(0, 1)
    return (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)



def plot_augmentations(sample: Path, mean, std, palette: np.ndarray, image_size: config.ImageSize | None = config.DEFAULT_IMAGE_SIZE, n_examples: int = 5, seed: int = 0,
                       augmentation: str = config.AUGMENTATION):
    """Show a sample and several random augmentations, with the matching masks.

    Use it to check by eye that image and mask stay aligned and that the
    augmentations look realistic. ``augmentation`` selects the profile to look at,
    e.g. "strong" for experiment D.
    """
    image, mask = load_sample(sample)
    if image_size is not None:
        image, mask = resize_pair(image, mask, image_size)
    transform = build_transforms(mean, std, train=True, seed=seed, augmentation=augmentation)

    fig, axes = plt.subplots(2, n_examples + 1, figsize=(3.2 * (n_examples + 1), 3.6))
    axes[0, 0].imshow(image)
    axes[1, 0].imshow(colorize(mask, palette))
    axes[0, 0].set_title("original")
    for i in range(n_examples):
        out = transform(image=image, mask=mask)
        axes[0, i + 1].imshow(denormalize(out["image"], mean, std))
        axes[1, i + 1].imshow(colorize(out["mask"].numpy(), palette))
        axes[0, i + 1].set_title(f"augmented {i + 1}")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    return fig