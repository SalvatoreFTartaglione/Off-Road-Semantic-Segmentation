# Dataset Analysis and Train/Validation Split

This document summarises what the exploratory data analysis revealed about the
off-road segmentation dataset, and how the
train/validation split was designed to address them.

## 1. Dataset overview

- 931 labelled frames, all 1024x544 RGB, extracted from a camera mounted on a moving car.
- Labels are class indices 0-8 stored in palette PNGs. Class 0 is background
  (ego-vehicle hood and unlabelled areas) and is excluded from the evaluation metric.
- The metric is the per-class IoU averaged over images, so every one of the 8
  semantic classes weighs the same, no matter how many pixels it covers.

| Label | Class | Label | Class |
|---|---|---|---|
| 1 | smooth trail | 5 | obstacle |
| 2 | traversable grass | 6 | non-traversable low vegetation |
| 3 | rough trail | 7 | high vegetation |
| 4 | puddle | 8 | sky |

The mapping was established by visual inspection of the colour
palette and of the spatial position of each class.

**Folders.** `data/train` holds these 931 frames, and the split described below is built from them
only. `data/test` holds a separate set of 145 labelled frames, used only to measure the final
network (see [prediction.md](prediction.md)): the training code reads `config.TRAIN_DIR` alone.

## 2. Findings from the analysis

### 2.1 Strong class imbalance

Share of labelled pixels (classes 1-8) in the training set:

| Class | Pixel share |
|---|---|
| high vegetation | 37.3% |
| smooth trail | 15.6% |
| rough trail | 15.6% |
| traversable grass | 13.6% |
| sky | 10.5% |
| non-traversable low vegetation | 6.3% |
| obstacle | 0.95% |
| puddle | 0.21% |

Puddle and obstacle are rare in pixels and sparse in space: their frequency
never exceeds 1% and 2.9% respectively at any image position. Since the metric
weights all classes equally, a model that ignores them could still reach a high
pixel accuracy while scoring poorly. Pixel accuracy is therefore not a reliable
indicator, and the loss function and the split must take the rare classes into account.

### 2.2 Label noise on the puddle class

Inspecting the images with the largest puddle area showed that annotators
sometimes marked a whole wet stretch of road as puddle. This is annotation
noise inherent to the dataset. It cannot be corrected because the data may not
be modified, so it is reported as a known limitation that caps the achievable
IoU on this class.

### 2.3 Spatial structure of the scenes

The per-class spatial priors show a very regular layout: sky always at the top,
a band of high vegetation around the horizon, trails converging towards the
centre of the lower half. Consequences for augmentation:

- Vertical flips are not valid, since they produce scenes that never occur.
- Horizontal flips are safe, because the layout is left-right symmetric.

### 2.4 Variable illumination

The images are taken under different daylight conditions (from overcast and
dark to bright sun), so the brightness distribution is wide. This motivates
photometric augmentation (brightness and contrast).

### 2.5 Temporal correlation between frames (leakage risk)

The frames come from a moving camera, so frames with nearby IDs show almost the
same scene. This was quantified by computing the mean absolute difference between
grayscale thumbnails of frames at a given ID distance k:

| Frame pairs | Mean pixel difference |
|---|---|
| consecutive frames (k = 1) | 36.0 |
| random pairs | 53.7 |

The difference grows with k and reaches the level of random pairs at about
**k = 5** (within 10% of the random level), saturating around k = 10.
Frames closer than a few IDs are therefore near duplicates.

In addition, the plot of the difference between each frame and the next one
along the ID order shows two patterns:

- sharp peaks (up to about 120) that correspond to **scene changes**;
- a **final stretch of roughly 100 frames** where the difference falls to
  12-20, against an average of 36. These frames are almost identical to each
  other, probably because the vehicle was moving slowly or was stationary.

Note that the sample names reach ID 1196 although only 931 samples exist, so
some IDs are missing. All "distances" in this analysis are positions in the
sorted list, not differences of IDs.

## 3. The problem

A random split has two failure modes on this data:

1. **Leakage.** Near-duplicate frames end up in both training and validation.
   The model is then validated on scenes it has effectively already seen, and the
   validation score overestimates the performance on the private test set.
   The near-identical final stretch makes this especially severe.
2. **Unrepresentative validation.** With puddle present in only about 4% of the
   images, a random split can leave very few, or no, puddle examples in
   validation, making the IoU on that class impossible to measure.

## 4. The solution: block-based, class-balanced split with purging

The split is built in four steps (`src/split.py`).

### Step 1: adaptive contiguous blocks

Frames are scanned in ID order and grouped into contiguous blocks. A block is
closed only when both conditions hold:

- it contains at least `min_block_size = 30` frames;
- the difference between the current frame and the next one is above the median of
  all consecutive differences, that is, the cut happens at a scene change.

Runs of near-duplicate frames are never cut, unless a block reaches
`max_block_size = 120` frames. A tiny trailing block is merged into the previous one.
Cutting at scene changes keeps each block coherent, and not cutting inside
near-identical runs prevents the final stretch from being split across train and validation.

### Step 2: assigning whole blocks to train or validation

Each block goes entirely to one set. The assignment minimises the following cost,
where `f` is the target validation fraction (0.2), computed over classes 1-8:

```text
cost = mean_c |pixel_share_val(c)  - f| / f
     + mean_c |image_share_val(c)  - f| / f
     +        |frame_share_val     - f| / f
     + 10 if any class is missing from the validation images
```

- `pixel_share_val(c)` is the fraction of the pixels of class c that lies in validation.
- `image_share_val(c)` is the fraction of the images containing class c that lies in validation.

Balancing both pixel and image counts matters: a class can be rare because it
appears in few images, or because it covers few pixels when it appears, and the
split should handle both.

The search is a randomised local search. Each of 300 trials starts from a random
set of validation blocks and repeatedly swaps one validation block with one
training block while the cost decreases. The best trial is kept. A fixed random
seed makes the result reproducible.

### Step 3: purging the boundaries

Even with whole blocks, the last training frame before a validation block is a
neighbour of the first validation frame. Training frames closer than
`purge_gap = 5` IDs (the decorrelation distance measured in Section 2.5) to any
validation frame are excluded from training. This technique is standard for
time-series cross-validation. It only removes frames from the provided data, so
it does not extend the training set.

### Step 4: persistence

The split is saved to `data/splits/split_seed42.json`, with its parameters and
the names of the training, validation and purged samples, so that every
experiment uses exactly the same protocol.

## 5. Result

| Set | Frames |
|---|---|
| training | 690 |
| validation | 186 (20%) |
| purged (unused) | 55 |

- The dataset is divided into 27 blocks of 30-38 frames, plus one block of 113
  frames corresponding to the near-duplicate final stretch, which is kept whole.
- The class distribution, in both pixels and images, closely matches between
  training and validation for all eight classes.
- Puddle appears in 33 training and 8 validation images.
- The minimum distance between a training frame and a validation frame is
  6 positions, larger than the purge gap.

## 6. Limitations

- Eight puddle images in validation give a noisy IoU estimate for that class.
  Combined with label noise, results on it should be interpreted with caution.
  A block-wise k-fold cross-validation can be used later for more stable estimates.
- The decorrelation distance is estimated with a simple measure (mean difference
  of grayscale thumbnails). It captures global scene similarity, not exact
  semantic overlap.
- Missing IDs in the sample list mean that a few neighbours in the sorted list
  are further apart in time than their position suggests. This only makes the
  split more conservative.

## 7. Reproducibility

```python
from src.split import make_split, save_split

split = make_split(names, stats["per_image"], cons, val_fraction=0.2, seed=42)
save_split(split, "data/splits/split_seed42.json")
```