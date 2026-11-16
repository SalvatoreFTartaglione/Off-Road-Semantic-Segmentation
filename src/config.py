from pathlib import Path


# SPLIT
VAL_SPLIT = 0.20
MIN_BLOCK_SIZE = 30
MAX_BLOCK_SIZE = 120
PURGE_GAP = 5
SEED = 42

# DATALOADERS
NUM_WORKERS = 4
BATCH_SIZE = 16
ImageSize = tuple[int, int]  # (height, width)
DEFAULT_IMAGE_SIZE: ImageSize = (272, 512) # half of the original 544x1024


# AUGMENTATION
# Every profile lists the parameters of the random transforms applied online to the training images
# (a new random version of every image at every epoch). Geometric transforms act on image and mask,
# photometric ones on the image only. A transform with probability 0 is not added to the pipeline.
AUGMENTATION = "standard"           # profile used by the experiments that do not choose one
AUGMENTATIONS = {
    "standard": {                   # used by experiments A, B, C and E
        "flip_p": 0.5,              # horizontal flip
        "affine_p": 0.6,            # random zoom + small rotation + shift, together
        "scale": (0.85, 1.25),      # zoom factor range (below 1 zooms out, the border is labelled background)
        "rotate": (-5, 5),          # degrees
        "translate": (-0.05, 0.05), # shift as a fraction of the image size
        "brightness_contrast_p": 0.7,
        "brightness_limit": 0.3,
        "contrast_limit": 0.3,
        "gamma_p": 0.3,
        "gamma_limit": (75, 130),   # in percent, 100 = unchanged
        "saturation_p": 0.3,
        "saturation": 0.2,          # the hue is never changed: the colour identifies the classes
        "blur_p": 0.0,
        "blur_sigma": (0.5, 1.5),   # standard deviation of the Gaussian kernel, in pixels
        "noise_p": 0.0,
        "noise_std": (0.02, 0.08),  # standard deviation of the Gaussian noise, as a fraction of the intensity range
        "dropout_p": 0.0,
        "dropout_holes": (1, 4),    # number of rectangular holes
        "dropout_size": (0.05, 0.25),   # side of every hole, as a fraction of the image side
    },
    "strong": {                     # used by experiment D: wider ranges and three extra transforms
        "flip_p": 0.5,
        "affine_p": 0.8,
        "scale": (0.7, 1.5),
        "rotate": (-8, 8),
        "translate": (-0.1, 0.1),
        "brightness_contrast_p": 0.8,
        "brightness_limit": 0.4,
        "contrast_limit": 0.4,
        "gamma_p": 0.5,
        "gamma_limit": (60, 150),
        "saturation_p": 0.5,
        "saturation": 0.35,
        "blur_p": 0.2,
        "blur_sigma": (0.5, 1.5),
        "noise_p": 0.2,
        "noise_std": (0.02, 0.08),
        "dropout_p": 0.3,           # holes are black in the image, the mask is left untouched
        "dropout_holes": (1, 4),
        "dropout_size": (0.05, 0.25),
    },
}


# LOSS
LOSS_CE_WEIGHT = 1.0        # weight of the cross-entropy term
LOSS_DICE_WEIGHT = 1.0      # weight of the Dice term
CLASS_WEIGHT_POWER = 0.5    # class weight = (f_max / f_c) ** power, 0 = uniform, 1 = full inverse frequency
MAX_CLASS_WEIGHT = 20.0     # cap on the class weights, for stability
BACKGROUND_WEIGHT = 1.0     # cross-entropy weight of class 0, on the same scale as the other classes
DICE_SMOOTH = 1.0           # smoothing term of the Dice loss
IGNORE_BACKGROUND = False   # if True, pixels with label 0 are ignored by both terms


# MODEL
MODEL_NAME = "unet"         # architecture to build
UNET_BASE_CHANNELS = 16     # channels of the first level, doubled at every downsampling
UNET_DEPTH = 4              # number of downsampling steps (inputs are padded to a multiple of 2 ** depth)


# PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"              # the provided training samples: the only ones used to train and validate
TEST_DIR = DATA_DIR / "test"                # labelled samples kept for the final test, never used for training
SPLIT_PATH = DATA_DIR / "splits" / "split_seed42.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs"       # one sub-folder per experiment


# EXPERIMENTS
EXPERIMENT = "D"            # experiment run by train(), unless another one is passed explicitly
# Besides the keys below an experiment may set "epochs", "weight_decay" and "augmentation": the ones it
# does not set fall back to EPOCHS, WEIGHT_DECAY and AUGMENTATION (so A, B, C and E need no overrides).
# It may also set "final_training": True (default False) to train on all the frames, see experiment F.
EXPERIMENTS = {
    "A": {
        "description": "U-Net written from scratch, random weights",
        "model": "unet",
        "model_kwargs": {},                 # base channels and depth come from UNET_* above
        "normalization": "dataset",         # mean/std computed on the training samples
        "encoder_lr_factor": 1.0,           # learning rate of the encoder relative to the decoder
    },
    "B": {
        "description": "ResNet18 encoder pre-trained on ImageNet + U-Net decoder",
        "model": "resnet18_unet",
        "model_kwargs": {"pretrained": True},
        "normalization": "imagenet",        # the statistics the pre-trained encoder was trained with
        "encoder_lr_factor": 0.1,           # fine-tune the pre-trained encoder gently
    },
    "C": {
        "description": "Same network as B with random weights (isolates the effect of pre-training)",
        "model": "resnet18_unet",
        "model_kwargs": {"pretrained": False},
        "normalization": "imagenet",        # same input as B, so that only the weights differ
        "encoder_lr_factor": 1.0,
    },
    "D": {
        "description": "Same network and schedule as B with stronger regularisation: strong augmentation and higher weight decay",
        "model": "resnet18_unet",
        "model_kwargs": {"pretrained": True},
        "normalization": "imagenet",
        "encoder_lr_factor": 0.1,
        "augmentation": "strong",           # see AUGMENTATIONS
        "weight_decay": 5e-2,               # 500 times the value of A, B, C and E
        "epochs": 110,                      # same as B, so that the schedule is not a difference between B and D
    },
    "E": {
        "description": "Same recipe as B with a deeper pre-trained encoder: ResNet50 instead of ResNet18",
        "model": "resnet50_unet",
        "model_kwargs": {"pretrained": True},
        "normalization": "imagenet",        # the statistics of the pre-trained ResNet50 are the same as ResNet18
        "encoder_lr_factor": 0.1,           # as in B: gentle fine-tuning of the pre-trained encoder
    },
    "F": {
        "description": "Final network: the recipe of B trained on every labelled frame, last epoch kept",
        "model": "resnet18_unet",
        "model_kwargs": {"pretrained": True},
        "normalization": "imagenet",
        "encoder_lr_factor": 0.1,
        # Train on all the frames (train + validation + the frames purged from the split), with no early
        # stopping and no choice of the best epoch: the weights of the last epoch are kept in final.pt.
        # The score printed at every epoch is then measured on frames the network is trained on.
        "final_training": True,
    },
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# TRAINING
EPOCHS = 110
LEARNING_RATE = 1e-3                # of the decoder (and of the whole network if encoder_lr_factor is 1)
WEIGHT_DECAY = 1e-4                 # AdamW
WARMUP_EPOCHS = 3                   # linear increase of the learning rate at the beginning
MIN_LR_FACTOR = 0.01                # the cosine schedule ends at LEARNING_RATE * MIN_LR_FACTOR
GRAD_CLIP_NORM = 1.0                # maximum norm of the gradient, None disables the clipping
EARLY_STOPPING_PATIENCE = 25        # epochs without improvement of the validation score, None disables it
USE_AMP = True                      # mixed precision (fp16) on the GPU

# PREDICTION
FINAL_WEIGHTS = OUTPUT_DIR / "experiment_F" / "final.pt"   # the network to deliver (experiment F)
PREDICT_BATCH_SIZE = 8              # images per forward pass inside predict (bounds the GPU memory at test time)
POSTPROCESS_MIN_AREA = 0.02         # a class that covers less than this fraction of an image is removed from it, 0 disables
POSTPROCESS_RARE_MIN_AREA = 0.0     # same threshold for the rare classes below (0: they are never removed)
RARE_CLASSES = (4, 5)               # puddle and obstacle: small by nature, a tiny region can be correct
