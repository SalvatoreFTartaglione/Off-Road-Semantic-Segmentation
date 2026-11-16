# Training Setup: Split, Data, Loss, Metric, Models and Training Procedure

This document describes how the models are trained and judged: how the data is split
and fed to the network, which loss function drives the optimisation, which metric is
used to judge the results and to choose the best epoch, which networks are compared,
and what exactly happens during a training run. The reasoning behind the
train/validation split is detailed in
[data_analysis_&_split.md](data_analysis_&_split.md); here only what matters for
training is summarised. How the final network predicts and how it scores on the test set is
described in [prediction.md](prediction.md).

All hyperparameters live in [`src/config.py`](../src/config.py), so every experiment
can be reproduced by changing a single file.

## 1. Overview

| Component | Module | Purpose |
|---|---|---|
| Train/validation split | `src/split.py` | Leakage-aware split, saved once to `data/splits/split_seed42.json` |
| Data pipeline | `src/dataset.py` | Resizing, augmentation, normalisation, dataloaders |
| Loss | `src/losses.py` | Class-weighted cross-entropy + Dice |
| Metric | `src/metrics.py` | Per-class IoU, identical to the official evaluation |
| Models | `src/models.py` | Hand-written U-Net and U-Net with a ResNet18 or ResNet50 encoder |
| Training | `src/train.py` | Training loop, checkpoints, experiment runner |

Hyperparameters used by these components:

| Parameter | Value | Used by |
|---|---|---|
| `VAL_SPLIT` | 0.20 | split |
| `MIN_BLOCK_SIZE` / `MAX_BLOCK_SIZE` | 30 / 120 | split |
| `PURGE_GAP` | 5 | split |
| `SEED` | 42 | split (the dataloaders use the same value, 42, as the default of their `seed` argument) |
| `DEFAULT_IMAGE_SIZE` | (272, 512) | dataset, loss |
| `BATCH_SIZE` | 16 | dataloaders |
| `NUM_WORKERS` | 4 | dataloaders |
| `LOSS_CE_WEIGHT` / `LOSS_DICE_WEIGHT` | 1.0 / 1.0 | loss |
| `CLASS_WEIGHT_POWER` | 0.5 | loss |
| `MAX_CLASS_WEIGHT` | 20.0 | loss |
| `BACKGROUND_WEIGHT` | 1.0 | loss |
| `DICE_SMOOTH` | 1.0 | loss |
| `IGNORE_BACKGROUND` | False | loss |
| `UNET_BASE_CHANNELS` / `UNET_DEPTH` | 16 / 4 | hand-written U-Net |
| `AUGMENTATION` | "standard" | augmentation profile of the experiments that do not choose one (Section 3.3) |
| `AUGMENTATIONS` | profiles `standard` and `strong` | parameters of every random transform (Section 3.3) |
| `EXPERIMENT` | "A" | which experiment `train()` runs by default (Section 6.3) |
| `EXPERIMENTS` | A, B, C, D, E, F | settings of every experiment, which may override `epochs`, `weight_decay` and `augmentation`, or set `final_training` (Sections 6.3 to 6.6) |
| `EPOCHS` | 110 | training, unless the experiment sets its own |
| `LEARNING_RATE` | 1e-3 | optimiser (decoder, or the whole network when `encoder_lr_factor` is 1) |
| `WEIGHT_DECAY` | 1e-4 | optimiser (AdamW), unless the experiment sets its own (D: 5e-2) |
| `WARMUP_EPOCHS` / `MIN_LR_FACTOR` | 3 / 0.01 | learning-rate schedule |
| `GRAD_CLIP_NORM` | 1.0 | gradient clipping |
| `EARLY_STOPPING_PATIENCE` | 25 | early stopping |
| `USE_AMP` | True | mixed precision |
| `IMAGENET_MEAN` / `IMAGENET_STD` | (0.485, 0.456, 0.406) / (0.229, 0.224, 0.225) | normalisation of experiments B, C, D, E and F |

## 2. Train/validation protocol

### 2.1 The split in short

The 931 frames come from a moving camera, so frames with close IDs are near
duplicates. A random split would put near duplicates in both sets and inflate the
validation score. The split is therefore built from **contiguous blocks of frames**,
cut at scene changes, assigned as a whole to train or validation while balancing the
class distribution, and surrounded by a **purge gap** of 5 frames that are not used
for training.

| Set | Frames |
|---|---|
| training | 690 |
| validation | 186 (20%) |
| purged (unused) | 55 |

Class balance between the two sets (share of labelled pixels, classes 1-8):

| Class | Train pixels % | Val pixels % | Train images | Val images |
|---|---|---|---|---|
| smooth trail | 15.64 | 16.61 | 427 | 101 |
| traversable grass | 13.57 | 13.66 | 484 | 117 |
| rough trail | 15.55 | 14.80 | 370 | 114 |
| puddle | 0.21 | 0.16 | 33 | 8 |
| obstacle | 0.95 | 0.89 | 118 | 33 |
| non-traversable low vegetation | 6.29 | 5.98 | 261 | 70 |
| high vegetation | 37.33 | 37.40 | 684 | 184 |
| sky | 10.46 | 10.49 | 644 | 153 |

### 2.2 How the split is used during training

- **The split is computed once** and stored as a JSON file with the parameters and the
  names of the training, validation and purged samples. Training code never recomputes
  it, it only loads it with `load_split`, so every experiment sees the same data.
- **The validation set is never used for fitting anything.** Not the network weights,
  and not the preprocessing: the normalisation statistics (Section 3.2) and the class
  weights of the loss (Section 4.2) are computed on the training samples only.
- **The validation score drives the decisions:** which epoch of a run is the best one
  (Section 8), when to stop a run that no longer improves, and which experiment and
  hyperparameters are better.
- **Final model (experiment F, Section 6.6):** once architecture and hyperparameters were
  fixed, the model was retrained on all the frames of `data/train` (training, validation and
  purged), because the block split removes about a fifth of the scenes from the training set.
  In that last run there is no validation score to monitor, so the number of epochs is taken
  from the previous runs and the weights of the last epoch are kept. It is evaluated on the
  separate test set (see [prediction.md](prediction.md)).

### 2.3 Why the validation score can be trusted

Because whole blocks are held out and the boundaries are purged, validation scenes are
not near duplicates of training scenes. If the network starts to memorise the training
scenes, the validation score stops improving or drops. With a random split the
validation frames would have near twins in the training set and the score would stay
high even while the model overfits.

## 3. Data pipeline

### 3.1 Resolution

Images and masks are resized from 1024x544 to **512x272** when they are loaded.
Images use area interpolation (the best choice when shrinking), masks use nearest
neighbour so that they never contain invented class indices. Half resolution makes
the images four times smaller in pixels, which speeds up training at the cost of some detail on small objects (puddle and
obstacle). The size is a parameter, so a
different resolution can be tested without touching the code.

### 3.2 Normalisation

Pixel values are scaled to [0, 1] and standardised per channel with the mean and
standard deviation of the **training set only**, computed at the working resolution:

| | R | G | B |
|---|---|---|---|
| mean | 0.427 | 0.446 | 0.422 |
| std | 0.261 | 0.260 | 0.262 |

The same values must be used at inference time, so they are saved together with the
trained weights.

### 3.3 Augmentation

Augmentations are applied to the training set only. Geometric transforms act on the
image and on the mask together, photometric transforms act on the image only. Each
choice follows from the data analysis.

| Transform | Parameters | Reason |
|---|---|---|
| Horizontal flip | p = 0.5 | Scenes are left-right symmetric |
| Affine: scale, rotation, translation | scale 0.85-1.25, rotation +-5 degrees, shift +-5%, p = 0.6 | Moderate changes of distance and camera pose. Rotation is small because the horizon is always horizontal |
| Brightness and contrast | +-30%, p = 0.7 | Illumination varies from overcast to bright sun |
| Gamma | 0.75-1.30, p = 0.3 | Simulates different exposures |
| Saturation | +-20%, p = 0.3 | Mild colour variation |

Deliberately **not** used:

- **Vertical flip:** the sky is always at the top, so flipped scenes never occur.
- **Hue shift:** colour identifies the classes (green vegetation, brown ground), so
  changing the hue would produce misleading examples.

Areas left empty by the affine transform are black in the image and labelled 0
(background) in the mask. The background is not scored by the metric, so padding does
not teach the network wrong classes.

The augmentation is **online**: the images are stored once, unchanged, and every time the
dataloader reads one it applies a new random version of the transforms. The network
therefore never sees exactly the same training image twice, and nothing is saved on disk.

**Profiles.** The parameters live in `config.AUGMENTATIONS`, one entry per profile, and an
experiment selects one by name. The table above is the `standard` profile (experiments A, B,
C and E). The `strong` profile (experiment D) widens the ranges (scale 0.7-1.5, rotation
+-8 degrees, shift +-10%, brightness/contrast +-40%, gamma 0.6-1.5, saturation +-35%) and adds
three transforms that are off in `standard`:

| Transform | Parameters | Reason |
|---|---|---|
| Gaussian blur | sigma 0.5-1.5 px, p = 0.2 | Robustness to motion blur and defocus |
| Gaussian noise | std 2-8% of the range, p = 0.2 | Robustness to sensor noise |
| Coarse dropout | 1-4 black rectangles, side 5-25% of the image, p = 0.3 | The mask is left untouched, so the network must predict the right label under the hole from the context |

A transform with probability 0 is not added to the pipeline at all, which keeps the
`standard` pipeline (and its random stream) identical to the one A, B, C and E were trained
with. A regression test guards this.

### 3.4 Dataloaders

- The resized samples are cached in RAM, so JPEG decoding does not slow every epoch.
- The training loader shuffles and drops the last incomplete batch, since a batch with
  a single image is a problem for batch normalisation. The validation loader does
  neither.
- Every worker process receives its own reproducible random seed for the augmentations,
  so workers do not apply identical transformations.

## 4. Loss function

### 4.1 The problem

The metric averages the IoU of the 8 semantic classes with the same weight, but the
pixels are extremely unbalanced: high vegetation covers 37% of them, puddle 0.2%.
A plain cross-entropy averages the error over all pixels, so a class covering 0.2% of
them contributes 0.2% of the loss. A network that never predicts puddle would lose very
little, yet it would lose an eighth of the final score. The loss must be aligned with
the metric.

### 4.2 Class-weighted cross-entropy

The cross-entropy of each pixel is multiplied by a weight that depends on its class:

```text
w_c = min( (f_max / f_c) ** power , max_weight )
```

where `f_c` is the pixel frequency of class c in the training set and `f_max` the
frequency of the most frequent semantic class (high vegetation), so that class has
weight 1 and every rarer class has a larger weight.

- `power = 0.5` is a compromise. With `power = 1` (full inverse frequency) puddle would
  weigh about 180 times more than high vegetation, which destabilises the training.
  With the square root the ratio is about 13.
- `max_weight = 20` caps the weights, so a class that is extremely rare or missing
  cannot make the loss explode.
- Class 0 (background) is not scored by the metric and gets the fixed `BACKGROUND_WEIGHT`
  of 1, on the same scale as the other classes.
- PyTorch divides the weighted sum by the sum of the weights of the target pixels, so
  the overall magnitude of the loss does not depend on the chosen weights.

Resulting weights on the training set at 512x272:

| Class | Weight |
|---|---|
| background | 1.00 |
| smooth trail | 1.55 |
| traversable grass | 1.66 |
| rough trail | 1.55 |
| puddle | 13.31 |
| obstacle | 6.28 |
| non-traversable low vegetation | 2.44 |
| high vegetation | 1.00 |
| sky | 1.89 |

### 4.3 Dice loss

The Dice coefficient of a class measures the overlap between prediction and ground
truth, regardless of how large the class is:

```text
dice_c = (2 * sum(p_c * g_c) + smooth) / (sum(p_c) + sum(g_c) + smooth)
loss   = 1 - mean over classes 1-8 of dice_c
```

`p_c` are the softmax probabilities of class c, so the loss is differentiable, `g_c`
is 1 where the ground truth is class c, and the sums run over every pixel of the batch.

- Every class counts the same in the average, as in the metric.
- It is closely related to the IoU: for a single class they are monotonically linked
  by `IoU = Dice / (2 - Dice)`.
- Predicting class c on a pixel whose label is another class increases `sum(p_c)` and
  lowers `dice_c`, so false positives are penalised.
- The Dice is averaged over classes 1-8 only, like the metric. Background pixels still
  take part in the sums, so predicting a semantic class on the car hood is penalised.

The ground truth is never expanded into a one-hot tensor, which would take
`batch x 9 x H x W` floats. The intersection is obtained by picking, for each pixel, the
probability of its true class (`gather`) and summing it per class (`scatter_add`).

### 4.4 Combination

```text
loss = LOSS_CE_WEIGHT * weighted_cross_entropy + LOSS_DICE_WEIGHT * dice_loss
```

The two terms complement each other. The cross-entropy gives stable gradients for every
pixel, which is important at the beginning of the training when the probabilities are
almost random. The Dice drives the overlap of every class, including the rare ones.
Both terms are computed in float32 even under mixed precision, because softmax and
sums over millions of pixels are numerically delicate in float16.

### 4.5 The background class

The background (car hood and unlabelled borders) is kept as a normal class. It might
seem natural to ignore it since it is not scored, but the metric computes the union of
each class as `(ground truth == c) | (prediction == c)`. A pixel with label 0 predicted
as class 3 is therefore a false positive for class 3 and lowers its IoU. The network
must learn where the background is. Setting `IGNORE_BACKGROUND = True` removes label-0
pixels from both terms, and the comparison of the two options is a planned experiment.

### 4.6 Usage

```python
from src.losses import create_loss

criterion = create_loss(train_samples, device="cuda")    # weights from training samples only
parts = criterion.components(logits, masks)              # {"total", "ce", "dice"}
parts["total"].backward()
```

The model must output raw logits of shape `(B, 9, H, W)`, without softmax.

### 4.7 Verification

Tested against reference implementations:

- the memory-efficient Dice equals a naive one-hot implementation, both with and without
  `ignore_background`;
- the cross-entropy term equals `torch.nn.functional.cross_entropy` with the same weights;
- a perfect prediction gives a loss of zero;
- with `ignore_background` the loss does not change when the logits on background pixels
  change, while without it the loss does change;
- loss and gradients are finite under mixed precision on the GPU, and the loss decreases
  when a small network is trained on one batch.

## 5. Evaluation metric

### 5.1 Definition

The metric is the following:

1. For every image and every semantic class c in 1-8, compute
   `IoU = |gt == c AND pred == c| / |gt == c OR pred == c|`.
2. If class c is absent from both ground truth and prediction, the union is empty and
   the value is NaN, which is ignored.
3. If class c is absent from the ground truth but predicted, the IoU is 0. Predictions
   of classes that are not there are penalised.
4. The **score** is the NaN-aware mean over the whole matrix of images x classes.

Note that the official score is the mean over all valid (image, class) entries, not the
mean of the per-class means. A class that appears in more images has more entries and
weighs more. Both quantities are reported:

| Quantity | Meaning |
|---|---|
| `score` | Official score. Used to select the best model |
| `mean_class_iou` | Mean of the per-class IoUs, each class weighing the same |
| `per_class` | Mean IoU of each class over the images |
| `per_image` | Mean IoU of each image over the classes, useful to find the worst samples |
| `iou_matrix` | Raw IoU of every image and class |

### 5.2 Implementation

The reference code (test.ipynb) loops over classes with NumPy on each image, which is slow. The
implementation in `src/metrics.py` computes the IoU of all classes and all images of a
batch at once. Every pixel is mapped to the index `image * 9 + class`, and a single
`bincount` gives the area of each class in the prediction, in the ground truth, and in
their intersection. The union is `pred_area + gt_area - intersection`. It works on GPU
tensors, accepts either logits `(B, 9, H, W)` or labels `(B, H, W)`, and accumulates
batches, so the whole validation set can be evaluated in a loop.

```python
from src.metrics import MeanIoU, metrics_report

metric = MeanIoU()
for images, masks in val_loader:
    metric.update(model(images), masks)
results = metric.compute()
print(results["score"])
metrics_report(results)
```

### 5.3 Per-class table

`metrics_report` returns the IoU of each class together with the number of images in
which that class counted (non-NaN entries). The second column matters when reading the
results: the IoU of a class that counts in few images is noisy.

## 6. Models and experiments

### 6.1 Experiment A: U-Net written from scratch

A U-Net (Ronneberger et al., 2015) implemented with plain PyTorch layers, with random
initial weights (Kaiming initialisation).

- **Building block** (`DoubleConv`): two times 3x3 convolution, batch normalisation and
  ReLU. The convolutions have no bias, because the batch normalisation that follows
  subtracts the batch mean, which cancels any constant added by a bias, and has its own
  learnable offset.
- **Encoder:** 4 levels with 16, 32, 64 and 128 channels. After each level a 2x2
  max-pooling halves height and width.
- **Bottleneck:** 256 channels at 1/16 of the input resolution.
- **Decoder:** 4 levels. Each one doubles the resolution with a transposed convolution,
  concatenates the output of the encoder level with the same resolution (skip connection)
  and applies a `DoubleConv`.
- **Head:** a 1x1 convolution produces 9 scores per pixel (raw logits, no softmax).
- Inputs of any size are accepted: the network pads them to a multiple of 16 and crops
  the output back.
- **1,942,713 trainable parameters.**

The skip connections matter for this dataset: the deep path knows what is in the scene
(trail, sky, vegetation) but has lost the exact position of the boundaries, while the
skips carry the high-resolution detail. This is especially important for the small
classes (puddle and obstacle cover well under 1% of the pixels).

### 6.2 Experiments B and C: U-Net with a ResNet18 encoder

The same encoder-decoder idea, but the encoder is the ResNet18 of torchvision: a stem
(7x7 convolution, stride 2) and four stages of residual blocks that reduce the
resolution to 1/4, 1/8, 1/16 and 1/32 with 64, 128, 256 and 512 channels.

- **Decoder:** five levels. Each one enlarges the deep map with a bilinear interpolation
  to the exact size of the skip, concatenates the skip and applies a `DoubleConv`. The
  deepest stage (1/32 of the resolution) is the starting point of the decoder. The skips
  come from the three previous ResNet stages (1/16, 1/8 and 1/4), from the stem (1/2) and,
  at the last level, from the normalised input image itself (full resolution), so the output
  can use exact pixel values for the sharp boundaries of small objects. Bilinear
  interpolation to the size of the skip works for any input size, since the ResNet rounds
  sizes up.
- **Head:** 1x1 convolution to 9 scores per pixel.
- **14,328,649 trainable parameters**, about 11.2 M in the encoder and about 3.2 M in the
  decoder.
- **Experiment B** starts from the ImageNet weights of the ResNet18 (torchvision
  `IMAGENET1K_V1`). The encoder already detects edges, textures and shapes before seeing
  any image of this dataset, which is valuable with only 690 training images. The decoder
  is always trained from scratch, since ImageNet has no segmentation task.
- **Experiment C** is exactly the same network with random weights.
- **Experiments D and E** use the same architecture (D) or the same architecture with a deeper
  encoder (E, Section 6.5).

### 6.3 What the five experiments compare

| Experiment | Network | Initial weights | Input normalisation | Encoder learning rate | Augmentation | Weight decay | Epochs |
|---|---|---|---|---|---|---|---|
| A | hand-written U-Net | random | mean/std of the training set | - (one learning rate) | standard | 1e-4 | 110 |
| B | ResNet18 + decoder | ImageNet (encoder), random (decoder) | ImageNet statistics | 0.1 x the decoder one | standard | 1e-4 | 110 |
| C | ResNet18 + decoder | random | ImageNet statistics | same as the decoder | standard | 1e-4 | 110 |
| D | ResNet18 + decoder | ImageNet (encoder), random (decoder) | ImageNet statistics | 0.1 x the decoder one | strong | 5e-2 | 110 |
| E | ResNet50 + decoder | ImageNet (encoder), random (decoder) | ImageNet statistics | 0.1 x the decoder one | standard | 1e-4 | 110 |

- **B against C** differ only in the initial weights (and in the encoder learning rate that
  goes with fine-tuning): any difference measures the effect of pre-training.
- **C against A** differ in the architecture, with both trained from scratch.
- **D against B** differ only in the regularisation (augmentation profile and weight decay):
  any difference measures the effect of regularising more (Section 6.4).
- **E against B** differ only in the encoder, ResNet50 instead of ResNet18: any difference
  measures the effect of a deeper pre-trained encoder (Section 6.5).
- B, C, D and E use the ImageNet statistics so that they receive the same input, and so that the
  pre-trained encoder sees the kind of values it was trained on.
- In B the encoder has a ten times smaller learning rate than the decoder: its weights are
  already good and only need a gentle adjustment, while the decoder starts from random.

An experiment may override `epochs`, `weight_decay` and `augmentation` in its entry of
`config.EXPERIMENTS`; what it does not set comes from the global `EPOCHS`, `WEIGHT_DECAY` and
`AUGMENTATION` (the function `resolve_settings` in `src/train.py` applies this rule), and an
explicit `--epochs` argument wins over both. Only D overrides something: A, B, C and E use the
global values. The resolved values of every new run are stored in its checkpoints and in its
`results.json` (Section 7.5).

The experiment is selected by the global flag `EXPERIMENT` in `config.py`, or by the
argument of the training function, without editing any code:

```python
from src.train import train, run_experiments

train()                              # the experiment set in config.EXPERIMENT
train("D")                           # a specific one
run_experiments(("B", "D"))          # some of them, one after the other
run_experiments()                    # every experiment defined in config.EXPERIMENTS (A to F)
```

From a terminal: `python -u -m src.train --experiment B`.

### 6.4 Experiment D: B with stronger regularisation

**Why.** A, B and C were all limited by overfitting, not by the capacity of the network. B
scores 0.87 on the training images (no augmentation) against 0.415 on the validation ones;
its validation loss is minimal around epoch 8 and then grows from 1.6 to 2.6 while the
training loss falls to 0.23. The training set holds only about 27 independent scenes
(Section 2), which the network memorises.

**What.** D is exactly B (same network, pre-trained encoder, ImageNet statistics, encoder learning
rate, 110 epochs) with two changes, declared in the `D` entry of `config.EXPERIMENTS`:

| Setting | A, B, C | D | Intent |
|---|---|---|---|
| Augmentation | `standard` | `strong` | More varied training images make memorisation harder |
| Weight decay (AdamW) | 1e-4 | 5e-2 | Penalises large weights. AdamW decay is decoupled: every step multiplies the weights by `1 - lr * wd`, so with a small `wd` it has almost no effect |

**Result.** The regularisation did what it was meant to do on the losses but not on the metric:

| | B | D |
|---|---|---|
| Best validation score | 0.4152 (epoch 96) | 0.4093 (epoch 110) |
| Training loss at the best epoch | 0.226 | 0.432 |
| Validation loss at the best epoch | 2.609 | 2.071 |

D overfits less (smaller gap between training and validation loss, and the validation loss
rises less in the second half of the run), but its score is not better: the difference of 0.006
is within the noise of the validation set, and the per-class IoU shows no consistent winner
(D is slightly better on the ground classes and on puddle, B on obstacle, high vegetation and sky).
The conclusion is that memorising the training scenes is not the only thing that limits the
score: the other limits are the ambiguity of some labels, the small number of independent
scenes and the penalty that the metric gives to false positives on absent classes. D's best
epoch is the last one and its curve was still slowly rising, but not enough to change the
conclusion. B stays the reference, and no further regularisation experiments were run.

### 6.5 Experiment E: B with a ResNet50 encoder

**Why.** The only variation that has clearly moved the score so far is pre-training (B against
C, about +0.07). A deeper encoder, trained on the same ImageNet data, is the natural next step
in the same direction: its features are richer, and the decoder, which is trained from
scratch on 690 images, stays the same.

**What.** E is exactly B (augmentation `standard`, weight decay 1e-4, 110 epochs, encoder
learning rate 0.1 x the decoder one, ImageNet statistics) with one change: the encoder is the
ResNet50 of torchvision, pre-trained with the `IMAGENET1K_V2` weights (the more modern training
recipe of torchvision; they are downloaded the first time the experiment runs).

The decoder is the same, with one difference that follows from the encoder: the ResNet50 uses
"bottleneck" residual blocks, whose output has four times more channels in the four stages
(256, 512, 1024 and 2048 instead of 64, 128, 256 and 512). The skip connections are therefore
wider, and the first decoder block, which concatenates the 2048 channels of the deepest stage
with the 1024 of the stage above, is the widest (3072 input channels). The stem keeps 64
channels, and the resolutions of all the stages are those of the ResNet18 (Section 6.2).

In the code, `ResNetUNet` takes a `backbone` argument (a key of `RESNET_BACKBONES`, which lists
the weights and the channels of every encoder), and `ResNet50UNet` is the same class with
`backbone="resnet50"`, registered as `"resnet50_unet"`. The ResNet18 networks are unchanged:
the checkpoints of B, C and D still load, and a test guards the parameter count (14,328,649).

**Cost.** Measured with random weights and synthetic inputs (the cost does not depend on the
values), batch size 16 at 512x272, mixed precision, on the laptop of the project:

| | B, C, D (ResNet18) | E (ResNet50) |
|---|---|---|
| Trainable parameters | 14.3 M | 32.5 M (encoder 23.5 M, decoder 9.0 M) |
| Training, per epoch (43 iterations, without validation) | about 32 s | about 83 s |
| Peak GPU memory in training | 3.04 GB | 3.78 GB |
| Inference at 1024x544, one image: speed, peak memory | 64 fps, 0.34 GB | 29 fps, 0.41 GB |
| Inference at 1024x544, batch of 8: speed, peak memory | 74 fps, 2.5 GB | 45 fps, 2.0 GB |

These numbers come from a short synthetic benchmark and they depend on the machine. In the real
run of 110 epochs (with the validation pass, the loss and the metric) E took 77 seconds per epoch
(B: 32) and its peak GPU memory was **4.60 GB** (B: 3.04 GB), higher than the benchmark and close
to the 5 GB limit of the project for training. Inference is far below the 4 GB limit, but E is
about 2.2 times slower at test time (one image at a time), which matters because the evaluation
also considers the frames per second.

**Result.** E is not clearly better than B:

| | B | E |
|---|---|---|
| Best validation score | 0.4152 (epoch 96) | 0.4233 (epoch 88) |
| Training loss at the best epoch | 0.226 | 0.212 |
| Validation loss at the best epoch | 2.609 | 2.598 |
| Training time per epoch, peak GPU memory | 31 s, 3.04 GB | 77 s, 4.60 GB |
| Inference speed at 1024x544 (benchmark) | 64 fps | 29 fps |

The difference of 0.008 in the score is below the noise of the validation set (about 0.01),
and the per-class IoU shows no consistent winner (E is slightly better on the ground classes,
puddle, obstacle and low vegetation, worse on traversable grass, high vegetation and sky). The deeper encoder fits the training set
better (lower training loss) and generalises no better: the limit of the score is not the
capacity of the encoder. E costs more than twice as much in training time and in test time, so
B stays the reference network. Both the regularisation (D) and the deeper encoder (E) leave the
score where it was, which points to the labels and to the small number of independent scenes as
the limit.

An ensemble, the average of the probabilities of several of these networks, was also tried on
the validation set (without any retraining). Plain argmax scores: B 0.4152, E 0.4233, B+E 0.4262,
B+D+E 0.4293. The gain over the best single network (E) is 0.003 for two networks and 0.006 for
three, at two to three times the cost at test time, so it is not worth it.

Run it with `python -u -m src.train --experiment E`. To save time the number of epochs can be
reduced, for example `--epochs 60`: B was already close to 0.40 around epoch 40 and its best epoch came late
only because of the slow cosine decay. A shorter run, however, has a different schedule and is
not exactly comparable with B.

Run it with `python -u -m src.train --experiment E`. To save time the number of epochs can be
reduced, for example `--epochs 60`: B was already close to 0.40 around epoch 40 and its best epoch came late
only because of the slow cosine decay. A shorter run, however, has a different schedule and is
not exactly comparable with B.


### 6.6 Experiment F: the final network

**Why.** Experiments A to E have chosen the recipe: B, the ResNet18 U-Net with a pre-trained
encoder, which matches the deeper ResNet50 (E) at a fraction of the cost and is not improved
by stronger regularisation (D). The network to deliver should learn from all the labelled
frames, not only from the 690 of the training split: the provided training set (`data/train`)
has 931 frames, of which 690 are in the training set, 186 in the validation set and 55 were
purged (Section 2.1), and with no validation left to protect from leakage the purged frames can
be used as well. That is 35% more frames. Neighbouring frames are near duplicates, so the gain in
independent scenes is smaller, but the validation frames come from blocks of road that the
network has never seen, which is what is missing from the training set.

**The test frames are never used.** The 145 labelled frames in `data/test` are kept for the final
test of the prediction function and are not part of the provided training set. The training code
reads only `config.TRAIN_DIR` (`data/train`), and `load_split` refuses a list of samples with
repeated names, because the folders `data/train` and `data/test` both contain samples called
`0000`, `0001`, and so on, and a dictionary of names would silently keep only one of each.

**What.** F is exactly B (same network, ImageNet weights, ImageNet statistics, augmentation
`standard`, weight decay 1e-4, 110 epochs, same learning-rate schedule) with the following
differences, all declared with `"final_training": True` in the `F` entry of `config.EXPERIMENTS`:

| | B | F |
|---|---|---|
| Training frames | 690 | all 931 (training, validation and purged) |
| Class weights of the loss | computed on the 690 frames | computed on the 931 frames |
| Best epoch and early stopping | yes, by validation score | **no**: nothing is held out |
| Saved weights | `best.pt`, the best epoch | `final.pt`, the **last** epoch |
| Iterations per epoch | 43 | 58 |

**Why the last epoch.** With no held-out data there is no score to choose an epoch with. The
last epoch is a good choice because the cosine schedule ends with a very small learning rate
(1e-5) and B's validation score is flat at the end: between epochs 95 and 110 it ranged from 0.410
to 0.415 (mean 0.413), and the last epoch scored 0.4141. The best score of B (0.4152) is the maximum
over about a hundred noisy epochs and is slightly optimistic (Section 8.3), so **the expected quality
of the last epoch is about 0.41**, with the post-processing of the prediction on top.

**What can and cannot be checked.** The score and the losses printed at every epoch are measured
on the 186 validation frames, which are now part of the training set: the log calls them
"seen-frames" values. They are a sanity check (they must fall, or rise, steadily, and end
clearly above the validation score of B, whose training-set score was about 0.87) and **not** an
estimate of the quality of the network. That estimate is the validation score of B. For the
same reason F is marked `trained on = train+val` in `compare_experiments` and is left out of
`plot_comparison`.

**Cost.** About 45 seconds per epoch (58 iterations of about 0.75 s plus the pass on the
validation frames, estimated from a two-epoch run), so 110 epochs take roughly 1.4 hours, and the
peak GPU memory is 3.04 GB, as for B.

Run it with:

```bash
python -u -m src.train --experiment F
```

It writes `outputs/experiment_F/`: `final.pt` (the weights to deliver, with the same content as a
`best.pt`: model name, normalisation, image size and settings, so that `load_checkpoint` and the
prediction function work the same), `last.pt` (full state, to resume an interrupted run),
`history.json` and `results.json`. As for the other experiments, one run uses one seed, and the
result cannot be verified on held-out data: if the log looks wrong (for instance the seen-frames
score stays near 0.4), the cause is a bug or an interrupted run, not a quality problem to be
fixed by training longer.

## 7. Training procedure

### 7.1 One epoch

Each epoch performs the following steps (`fit` in `src/train.py`):

1. **Training pass** over all the batches of the training set (43 batches of 16 images):
   forward pass, loss, backward pass, optimiser step (details in Section 7.2).
2. **Validation pass** over the validation set (12 batches), with the network in
   evaluation mode (batch normalisation uses its running statistics) and no gradients.
   It computes the loss and the metric of Section 5 on all 186 validation images.
3. **Learning-rate update** (Section 7.3).
4. **Logging:** one line with learning rate, training loss (and its two terms), validation
   loss, validation score and best score so far. An asterisk marks an epoch that improved
   the best score.
5. **Checkpoints:** `best.pt` if the validation score improved, `last.pt` always, plus the
   history of all the epochs (Section 7.5).
6. **Early stopping check** (Section 7.4).

### 7.2 What happens inside a training step

- **Optimiser:** AdamW, learning rate 1e-3, weight decay 1e-4 (5e-2 in experiment D).
- **Mixed precision:** the forward pass runs in float16 where it is numerically safe,
  which is faster and uses less GPU memory. The loss is computed in float32, because the
  softmax and the sums over millions of pixels are delicate in float16. A gradient scaler
  multiplies the loss before the backward pass, so that small float16 gradients do not
  underflow to zero.
- **Gradient clipping:** after unscaling, the norm of the gradient is limited to 1.0, which
  prevents occasional huge updates, especially in the first epochs when the decoder is random.
- **Memory layout:** model and inputs use the `channels_last` layout, which is faster on
  GPUs with tensor cores.

### 7.3 Learning-rate schedule

The learning rate is multiplied by a factor that depends on the epoch `e` (starting at 0)
and on the total number of epochs `E`:

```text
e < 3:   factor = (e + 1) / 3                                      (linear warmup)
e >= 3:  factor = 0.01 + 0.99 * 0.5 * (1 + cos(pi * (e - 3) / (E - 3)))   (cosine decay)
```

- The **warmup** (a third, two thirds, then the full learning rate) avoids damaging
  updates while the decoder is still random.
- The **cosine decay** lowers the learning rate smoothly down to 1% of its value, so that the
  network settles into a minimum at the end of the run. With 110 epochs the learning rate is
  1.0e-3 from epoch 4 and 1.0e-5 at epoch 110.
- Both parameter groups of experiments B, D and E follow the same factor, so the encoder keeps its
  ten times smaller rate throughout.
- Since the schedule depends on `E`, the total number of epochs must be decided before the
  run (see Section 9 for what happens when a run is extended).

### 7.4 Early stopping

The run stops if the validation score has not improved for 25 consecutive epochs. The
criterion is the same score used to select the best epoch (Section 8). With a cosine
schedule the best epochs are usually near the end, when the learning rate is small, so the
patience is deliberately generous.

### 7.5 Files produced by a run

Everything is written to `outputs/experiment_<name>/`, which is not versioned.

| File | Content |
|---|---|
| `best.pt` (`final.pt` for experiment F) | Weights of the epoch with the best validation score (for F, of the last epoch, see Section 6.6), and everything needed for inference: model name and arguments, mean and std of the normalisation, image size, epoch, score and per-class IoU, plus the resolved settings of the run (epochs, weight decay, augmentation profile, encoder learning-rate factor). It is the file to deliver. |
| `last.pt` | Full state of the last epoch: weights, optimiser, scheduler, gradient scaler, history, best score. It is used to resume a run. |
| `history.json` | For every epoch: learning rate, training loss (total, cross-entropy, Dice), validation loss (same three terms), validation score, mean class IoU, IoU of each class, seconds, peak GPU memory. |
| `results.json` | Summary of the run: resolved settings, best score and epoch, per-class IoU at the best epoch, number of parameters, seconds per epoch, peak GPU memory. Runs made before the settings were stored (A, B and C) lack the `settings` and `model` keys, and the comparison table then reads them from `config.EXPERIMENTS`. |

`load_checkpoint` rebuilds a model from a checkpoint without downloading the pre-trained
weights again (they come from the checkpoint), so inference works offline.

### 7.6 Measured cost

Measured in a short run (2 epochs) on the laptop of the project (RTX 3060 Laptop GPU with
6 GB), batch size 16, images at 512x272, mixed precision:

| Experiment | Parameters | Seconds per epoch | Peak GPU memory |
|---|---|---|---|
| A | 1.9 M | about 29 | 3.1 GB |
| B | 14.3 M | about 31 | 3.0 GB |
| C | 14.3 M | about 32 | 3.0 GB |
| D | 14.3 M | about 32 (the augmentation runs in the loader processes) | 3.0 GB |
| E | 32.5 M | about 77 | 4.6 GB |

The first epoch takes about 70 seconds, because of the start of the loader processes and
of the search of the fastest cuDNN kernels. At about 30 to 35 seconds per epoch, 110 epochs
take roughly one hour per experiment (about 2.4 hours for E, whose cost is detailed in
Section 6.5). All the peaks are below the 5 GB limit of the project
for training. The difference in peak memory between B and C, which are the same network,
was not investigated; it probably comes from the choice of cuDNN algorithms.

## 8. Monitored metrics and choice of the best epoch

### 8.1 What is recorded at every epoch

| Quantity | Where it is computed | Use |
|---|---|---|
| Training loss (total, cross-entropy, Dice) | training pass, averaged over the images | Is the network learning? |
| Validation loss (same three terms) | validation pass | Is it overfitting? A validation loss that rises while the training loss falls signals overfitting |
| **Validation score** | `MeanIoU.compute()["score"]` | **Selection of the best epoch** and early stopping |
| Mean class IoU | mean of the per-class IoUs | Secondary check: every class weighs the same |
| IoU of each class | `per_class` | Diagnosis: which classes are learned and which are not |
| Learning rate, seconds, GPU memory | the loop | Control of the run |

The losses are those of Section 4, with the class weights of the training set, in both the
training and the validation pass, so the two curves are comparable.

### 8.2 The selection metric

**The best epoch is the one with the highest validation score**, where the score is the
official metric of the course (Section 5): the NaN-aware mean of the IoU over all the
(image, class) pairs of the validation set, for the 8 semantic classes. An epoch replaces
the current best only if its score is strictly higher, so in case of a tie the earlier
epoch is kept. The checkpoint `best.pt` is overwritten when this happens, and the
comparison between experiments uses the best score of each one.

The reasons for choosing this metric:

- It is exactly what the private test set will be evaluated with, so it measures the right
  thing.
- It is not the loss. The loss is a proxy used to optimise the network (a class-weighted
  cross-entropy over the nine labels plus a Dice computed on the whole batch), and the epoch
  with the lowest validation loss is not necessarily the one with the best IoU.
- It is not the pixel accuracy, which would be dominated by the frequent classes and would
  hide the failures on puddle and obstacle.

### 8.3 How to read the metrics

- **Score and mean class IoU** are close but not identical: the score counts each valid
  (image, class) pair once, so a class that appears in many images weighs more, while the
  mean class IoU gives every class the same weight. A model that improves one but not the
  other deserves a look at the per-class table.
- **Noise.** The validation set has 186 images, and only 8 contain puddle. Differences of a
  few thousandths between epochs, or between experiments, are within the noise and should
  not drive decisions.
- **Validation loss and score can disagree.** The loss of experiment D on the validation set is
  clearly lower than that of B and E (2.07 against 2.61 and 2.60), while its score is slightly
  lower (0.409 against 0.415 and 0.423). The reason is that the two measure different things.
  The score looks only at which class has the highest probability in every pixel. The
  cross-entropy looks at the probability given to the right class, and so it punishes a wrong
  pixel more the more confident the network was. On the validation set the three networks are
  wrong on about the same share of the pixels (accuracy 73.3% for B, 72.9% for D, 73.8% for E),
  and these wrong pixels account for 97 to 99% of the total cross-entropy. B and E are more
  confident when they are wrong (mean probability of the predicted class 0.898 and 0.910) than
  D (0.853), so their loss on those pixels is higher (5.8 and 6.0 against 4.3). D, trained with
  stronger augmentation and weight decay, is better calibrated, not better at segmenting. A
  rising validation loss together with a stable or rising score, as in B and E, is therefore
  overconfidence and not necessarily a sign that the segmentation gets worse. Only the score
  decides the best epoch (Section 8.2).
- **Selection bias.** The best epoch is chosen as the maximum over about a hundred epochs on
  the same validation set. The score of that epoch is therefore slightly optimistic, and the
  result on the private test set is expected to be somewhat lower.
- **Per-class IoU** at the best epoch is the main diagnostic for the report, together with
  the `images` column of the metric report (Section 5.4).

### 8.4 Comparing the experiments

```python
from src.train import compare_experiments, plot_comparison, plot_history, load_history

compare_experiments()                  # table, one row per experiment that has been run
plot_comparison()                      # score and losses of all the experiments on the same axes
plot_history(load_history("D"))        # detailed curves of one experiment
```

`compare_experiments` has the following columns:

| Column | Meaning |
|---|---|
| `score`, `best_epoch`, `epochs` | Best validation score, the epoch where it was reached, epochs actually run |
| `params (M)`, `s/epoch`, `gpu GB` | Cost of the experiment |
| `model`, `trained on`, `augmentation`, `weight decay`, `planned epochs` | What distinguishes the experiments (A to F): network, data (`train`, or `train+val` for the final network F, whose score is measured on frames it was trained on and is not comparable), augmentation profile, weight decay and planned epochs |
| `train loss`, `val loss` | The two losses at the best epoch: a large gap means overfitting, and the effect of the regularisation of D shows as a smaller gap |
| One column per class | IoU of the class at the best epoch |

`plot_comparison` draws the validation score, the validation loss and the training loss of
every experiment that has been run, so a regularised run can be compared with its baseline at a
glance: with less overfitting the validation loss stays lower and does not rise late in the run
while the training loss falls less. Experiments that have not been run are skipped by both.

## 9. Resuming and extending a run

### 9.1 Resuming an interrupted run

`last.pt` stores everything needed to continue: weights, optimiser state, scheduler, gradient
scaler, history, best score and best epoch. To resume (with the same total number of epochs
as the original run, because the learning-rate schedule depends on it):

```bash
python -u -m src.train --experiment B --resume
```

The run continues from the epoch after the last completed one, keeps appending to the same
history, and the best score and the early-stopping counter carry on from where they were.

### 9.2 Training more epochs after a run has finished

It is possible: the command is the same with a larger number of epochs, for example after a
run of 110 epochs:

```bash
python -u -m src.train --experiment B --epochs 150 --resume
```

The epoch counter continues from 111 to 150, `best.pt` keeps the best weights found so far
(it is only replaced if an extra epoch does better) and the history files are extended.

(Experiments that set their own number of epochs in `config.EXPERIMENTS` use it as the default
of `--epochs`, so a resume without the argument keeps the original schedule.)

The learning rate, however, does **not** continue smoothly. The schedule is a function of the
total number of epochs, so the new run follows the schedule of a 150-epoch run from epoch 111
on. The values for the first run of 110 epochs extended to 150 are:

| Epoch | Learning rate |
|---|---|
| 110 (last of the original run) | 1.0e-5 |
| 111 (first extra epoch, restored from the checkpoint) | 1.0e-5 |
| 112 | 1.7e-4 |
| 120 | 1.1e-4 |
| 130 | 5.9e-5 |
| 150 | 1.0e-5 |

The learning rate jumps up by more than a factor of ten after the first extra epoch and then
decays again: this is a warm restart. It may find a better minimum, or it may make the
validation score worse for a while. Nothing is lost, because the best epoch found before is
kept, but the extension is not guaranteed to help.

Two practical remarks:

- If the original run ended because of **early stopping**, it had already stopped improving
  for 25 epochs. Extending it stops again after a single extra epoch, since the counter of
  epochs without improvement carries over.
- It is cleaner to decide the number of epochs in advance. If the curves show that 110
  epochs were not enough (the validation score still rising at the end), a new run with a
  larger `EPOCHS` has a clean schedule and is comparable with the other experiments, which
  an extended run is not.

## 10. Limitations

- **Resolution gap.** Training and validation work at 512x272, while the official
  evaluation uses the original 1024x544 masks. The validation score is a good estimate
  but not identical to the official one. The final check will be done with the official
  notebook at full resolution.
- **Puddle.** The class has 8 validation images and, as noted in the data analysis, noisy
  labels. Its validation IoU has a large variance and should be read together with the
  `images` column of the metric report.
- **Class weights.** `power = 0.5` and `max_weight = 20` are reasonable defaults, not
  tuned values. The effect of the weighting and of the Dice term is to be measured with
  ablations (cross-entropy only, with and without weights, with and without Dice).
- **Single split.** All decisions are based on one validation set of 186 frames. A
  block-wise k-fold cross-validation would give a more stable estimate if the
  differences between models turn out to be small.
- **One run per experiment.** Every experiment is trained once, with one seed. Differences
  between experiments smaller than the variation between two runs of the same
  experiment cannot be resolved. Part of that variation comes from the non-deterministic
  choice of cuDNN kernels (`cudnn.benchmark`), which is used for speed.
- **Untuned hyperparameters.** Learning rate, weight decay, warmup length, the factor
  0.1 of the encoder learning rate in B, and the number of epochs are reasonable defaults
  that were not tuned. If B behaves oddly in the first epochs, the encoder learning rate is
  the first parameter to look at.
- **Optimistic best score.** The reported validation score is the maximum over the epochs
  (Section 8.3), so the test score is expected to be slightly lower.

## 11. Reproducibility

- Split: fixed `SEED`, saved in `data/splits/split_seed42.json`.
- Run: `train()` seeds Python, NumPy and PyTorch with `SEED`; the dataloader shuffling and the
  augmentation seeds of every worker derive from the `seed` argument of
  `create_dataloaders` (default 42).
- Statistics (normalisation and class weights): deterministic functions of the training
  samples listed in the split file.
- Hyperparameters: all in `src/config.py`, and the resolved settings of each new run (epochs,
  weight decay, augmentation profile, encoder learning rate) are stored inside its `best.pt`
  and `results.json`.
- Results are reproducible up to small variations caused by non-deterministic GPU kernels.

```python
from src.train import train

out = train("D")          # data, model, loss, optimiser and training loop, all from config.py
print(out["best_score"], out["best_epoch"])
```

## 12. Summary of the experiments

Validation score (official metric, 186 validation images) of the best epoch of every experiment, and
what each one showed. Differences below about 0.01 are within the noise of the validation set.

| Exp. | What changes with respect to B | Val. score | Best epoch | Conclusion |
|---|---|---|---|---|
| A | hand-written U-Net, random weights | 0.3504 | 10 | A network from scratch overfits after about 10 epochs |
| B | the reference: ResNet18 encoder pre-trained on ImageNet | 0.4152 | 96 | Pre-training is the variation that matters |
| C | B with random weights | 0.3458 | 10 | B against C: +0.069 from pre-training |
| D | strong augmentation and weight decay 5e-2 | 0.4093 | 110 | Less overfitting (lower validation loss), no better score |
| E | ResNet50 encoder | 0.4233 | 88 | Within noise of B, at twice the cost in training and in test |
| F | B on all the frames, last epoch kept | not measurable | 110 | The final network, scored on the test set |

The results that B, D and E share, and that point to the labels and to the number of independent
scenes as the limit of the score: the training score of B is 0.87 against 0.42 in validation, the
validation loss and the score disagree (Section 8.3), and an ensemble of the networks gains less
than 0.01 over the best single network. The only large and cheap gain found after the training is the post-processing of the
prediction (+0.036 on the validation set for B, +0.031 on the test set), described in
[prediction.md](prediction.md).
