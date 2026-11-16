# Prediction and Test Score

This document describes how the final network is used to predict and how it is scored on the
test set. The code is in [`src/predict.py`](../src/predict.py), the results are produced by
[`notebooks/03_test.ipynb`](../notebooks/03_test.ipynb), and the settings are in the
`PREDICTION` section of [`src/config.py`](../src/config.py).

## 1. The network and the test set

- **Network:** experiment F of [training.md](training.md) (Section 6.6): the ResNet18 U-Net with a
  pre-trained encoder, trained with the recipe of experiment B on all the labelled frames of
  `data/train`, last epoch kept. Weights: `outputs/experiment_F/final.pt`.
- **Test set:** the 145 labelled samples of `data/test`. They are never read by the training code
  (it uses only `config.TRAIN_DIR`), so the network has not seen them in any form.
  Because the network was trained with no validation set, **this is the only measurement of its
  quality**.

## 2. The `predict` function

```python
from src.predict import load_model, predict

model = load_model()                 # config.FINAL_WEIGHTS, on the GPU if there is one
labels = predict(model, image)       # image (H, W, 3) uint8   -> labels (H, W) uint8
labels = predict(model, batch)       # batch (B, H, W, 3) uint8 -> labels (B, H, W, 1) uint8
labels = predict(batch)              # the model is loaded the first time and reused
```

It works for any image size and any batch size, and performs all the processing itself
(the project requires it):

1. **Pre-processing.** Every image is resized to the training resolution, 512x272, with the same
   interpolation used in training (area), converted to float and normalised with the mean and
   standard deviation stored in the checkpoint (the ImageNet ones).
2. **Forward pass** in float16 on the GPU, in chunks of `PREDICT_BATCH_SIZE` (8) images, so that the
   memory does not grow with the size of the batch given to `predict`.
3. **Back to the original size.** The class scores are enlarged with a bilinear interpolation to the
   size of the input image and every pixel takes the class with the highest score. The output has
   the size of the input, as the labels have.
4. **Post-processing** (Section 3).

The checkpoint contains the model name, its arguments, the normalisation and the working
resolution, so `load_model` needs nothing else and never downloads the pre-trained weights.

## 3. Post-processing: removing very small classes

The metric (the mean IoU over all the pairs of image and class) is unforgiving of one kind of
error: when a class is **absent** from the ground truth of an image but the network predicts it on
a few pixels, the IoU of that pair is 0, and the 0 enters the average. If the class is absent
from both, the pair is ignored. On the validation set, 28% of the valid pairs of B and about a quarter of those of the other
networks were zeros of this kind, and without them the score would have been much higher.

The post-processing removes this noise. For every image, a class that covers less than
`POSTPROCESS_MIN_AREA` (2%) of the image is removed, and its pixels go to the best class among the
remaining ones. Details:

- the background is never removed;
- puddle and obstacle (`RARE_CLASSES`) are exempt (`POSTPROCESS_RARE_MIN_AREA` = 0), because they
  are small by nature and a small region of them can be right;
- the threshold is a fraction of the image, so it does not depend on the resolution;
- the areas are measured on the plain prediction of the network, and every class is judged
  independently of the others.

The threshold was chosen on the **validation set** of the experiments, where the score of the
networks was nearly constant for thresholds between 1% and 3% and decreased at 5%. It was
**not tuned on the test set**. The cost is speed: the post-processing makes the prediction about
one third slower (Section 5).

## 4. Evaluation

`evaluate` follows the evaluation code of the course: every image is given to `predict` alone, at
its original size (1024x544), the IoU of the classes 1 to 8 is computed against the label file, and
the score is the mean over all the pairs, ignoring those where the class is in neither map. The
metric is the same as in `src/metrics.py`, which works on the GPU during training, and a test
checks that the two agree.

Besides the score, `evaluate` reports the frames per second of the whole `predict` call
(preprocessing and post-processing included) and the peak GPU memory.

## 5. Results on the test set

| | Without post-processing | With post-processing |
|---|---|---|
| **Score (official)** | **0.4794** | **0.5106** |
| Mean class IoU | 0.3918 | 0.4092 |
| Pairs with IoU = 0 (valid pairs) | 249 (841) | 223 (783) |
| Frames per second (1024x544, one image at a time) | about 61-70 | about 47 |
| Peak GPU memory in `predict` | 0.14 GB | 0.14 GB |

Mean IoU per class with post-processing (images in which the class counts in brackets):
sky 0.835 (139), high vegetation 0.778 (145), traversable grass 0.478 (123), smooth trail 0.403 (113),
rough trail 0.306 (111), obstacle 0.281 (51), non-traversable low vegetation 0.193 (95),
puddle 0.000 (6).

How to read these numbers:

- The post-processing adds **0.031** to the score, in line with the gain measured on the validation
  set of the experiments, and it helps almost every class.
- The sky and the high vegetation are well segmented. The three kinds of ground (smooth trail,
  rough trail, traversable grass), the low vegetation and the obstacle are the weak classes:
  their labels are partly subjective and they are often confused with each other.
- **Puddle scores 0 on all 6 images** where it is present. It is the rarest class and the one with
  the noisiest labels (Section 3 of [data_analysis_&_split.md](data_analysis_&_split.md)). With only
  6 images, its contribution to the average is small but it is an obvious weakness.
- The test set has 145 images, so the score has an uncertainty of the order of 0.01-0.02.
  The difference with the validation score of the experiments (0.415 for B without post-processing) is
  a difference between two different sets of scenes, not an improvement of the network.
- The memory measured is the one allocated by PyTorch during the prediction, without the CUDA
  context (a few hundred MB more). It is far below the 4 GB of the test. In the measurement, the
  speed depends on the machine and on what else uses the GPU.

On the GPU, the same image gives a prediction that differs on about 0.002% of the pixels when it is
predicted alone or inside a batch, because the float16 kernels are not bit-identical across batch
sizes. On the CPU the two are identical (a test checks it).
