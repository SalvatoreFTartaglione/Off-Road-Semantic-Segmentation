"""Neural network architectures for semantic segmentation.

Every model maps an image batch (B, 3, H, W) to raw class scores ("logits")
(B, num_classes, H, W). No softmax is applied: the loss functions and the metric
take care of it.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from . import config
from .data import NUM_CLASSES
import torchvision
from torchvision.models import ResNet18_Weights, ResNet50_Weights

# The ResNets that can be used as encoder of ResNetUNet. For every one: the ImageNet weights to load,
# and the channels of the five feature maps used as skip connections (stem, layer1, ..., layer4).
# ResNet18 and ResNet34 use basic blocks (the channels of a stage are 64, 128, 256, 512), ResNet50
# and deeper use bottleneck blocks with 4 times more output channels in layer1 to layer4.
# The constructor is looked up by name when the model is built (torchvision.models.<name>).
RESNET_BACKBONES = {
    "resnet18": {"weights": ResNet18_Weights.IMAGENET1K_V1, "channels": (64, 64, 128, 256, 512)},
    "resnet50": {"weights": ResNet50_Weights.IMAGENET1K_V2, "channels": (64, 256, 512, 1024, 2048)},
}


class DoubleConv(nn.Sequential):
    """The basic building block of the U-Net: two times (3x3 convolution, BatchNorm, ReLU).
    - Conv 3x3 with padding 1: looks at a pixel and its 8 neighbours and keeps the
    height and width unchanged. Two of them in a row see a 5x5 neighbourhood.
    - BatchNorm: normalises every channel over the batch, which makes the training
    faster and more stable. It has its own learnable offset, so the convolution does
    not need a bias (bias=False).
    - ReLU: keeps positive values and zeroes the negative ones. Without a non-linearity,
    stacking convolutions would be equivalent to a single one.

    Only the number of channels changes: in_channels -> out_channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """
    Shape of the architecture for depth=4 and base_channels=b (the numbers in brackets
    are the height and width, H and W being the input ones)::

        input (3 channels)
          |
        [encoder 1]  DoubleConv   3 -> b          (H, W)       ----------------+ skip 1
          | MaxPool 2x2                                                         |
        [encoder 2]  DoubleConv   b -> 2b         (H/2, W/2)   ------------+ skip 2
          | MaxPool 2x2                                                     |   |
        [encoder 3]  DoubleConv  2b -> 4b         (H/4, W/4)   --------+ skip 3
          | MaxPool 2x2                                                 |   |   |
        [encoder 4]  DoubleConv  4b -> 8b         (H/8, W/8)   ----+ skip 4
          | MaxPool 2x2                                             |   |   |   |
        [bottleneck] DoubleConv  8b -> 16b        (H/16, W/16)      |   |   |   |
          | ConvTranspose 2x2 (16b -> 8b)         (H/8, W/8)        |   |   |   |
        [decoder 4]  concat with skip 4 (16b) -> DoubleConv -> 8b <-+   |   |   |
          | ConvTranspose 2x2 (8b -> 4b)          (H/4, W/4)            |   |   |
        [decoder 3]  concat with skip 3 (8b)  -> DoubleConv -> 4b <-----+   |   |
          | ConvTranspose 2x2 (4b -> 2b)          (H/2, W/2)                |   |
        [decoder 2]  concat with skip 2 (4b)  -> DoubleConv -> 2b <---------+   |
          | ConvTranspose 2x2 (2b -> b)           (H, W)                        |
        [decoder 1]  concat with skip 1 (2b)  -> DoubleConv -> b  <-------------+
          |
        [head]       Conv 1x1  b -> num_classes   (H, W)
          |
        logits (num_classes channels)
        
    Args:
        num_classes: number of output channels (9: background + 8 classes).
        in_channels: number of input channels (3 for RGB).
        base_channels: channels of the first level, doubled at every level.
        depth: number of downsampling steps. The input is padded internally so that
            its height and width are multiples of 2 ** depth.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, in_channels: int = 3,
                base_channels: int = config.UNET_BASE_CHANNELS, depth: int = config.UNET_DEPTH):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.depth = depth

        # Channels of every level: b, 2b, 4b, ..., 2^depth * b. The last one is the bottleneck.
        channels = [base_channels * 2 ** level for level in range(depth + 1)]

        # Encoder: one DoubleConv per level, from the full resolution down to H / 2^(depth-1)
        self.encoders = nn.ModuleList()
        previous = in_channels
        for level in range(depth):
            self.encoders.append(DoubleConv(previous, channels[level]))
            previous = channels[level]

        # The same pooling layer is used between all levels: it has no weights, it just keeps
        # the maximum of every 2x2 window, halving height and width
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Bottleneck: the deepest level, the smallest resolution (H / 2^depth) and most channels
        self.bottleneck = DoubleConv(channels[depth - 1], channels[depth])

        # Decoder: for every level, from the deepest to the first, an upsampling layer and a
        # DoubleConv. The upsampling halves the channels (c_level+1 -> c_level), the concatenation
        # with the skip connection doubles them again (2 * c_level), the DoubleConv brings
        # them back to c_level.
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in reversed(range(depth)):
            self.upsamples.append(
                nn.ConvTranspose2d(channels[level + 1], channels[level], kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(2 * channels[level], channels[level]))

        # Head: a 1x1 convolution is a per-pixel linear layer, from the b features of every pixel
        # to one score per class
        self.head = nn.Conv2d(channels[0], num_classes, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming initialisation, designed for ReLU networks: it keeps the scale of the
        activations stable across layers, so the training starts well."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # The output layer is not followed by a ReLU: it keeps PyTorch's default initialisation
        self.head.reset_parameters()

    def _pad_to_multiple(self, x: torch.Tensor) -> torch.Tensor:
        """Pad the bottom and right borders so that H and W are multiples of 2 ** depth.

        Every MaxPool halves the size (rounding down) and every transposed convolution doubles
        it, so with other sizes the decoder maps could not be concatenated with the skips.
        """
        multiple = 2 ** self.depth
        pad_height = (-x.shape[-2]) % multiple
        pad_width = (-x.shape[-1]) % multiple
        if pad_height == 0 and pad_width == 0:
            return x
        return F.pad(x, (0, pad_width, 0, pad_height), mode="replicate")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        x = self._pad_to_multiple(x)

        # 1. Encoder: extract features, remember each level's output (the skip connection)
        #    and halve the resolution
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        # 2. Bottleneck: the most abstract features, on the smallest map
        x = self.bottleneck(x)

        # 3. Decoder: go back up. The skips are consumed in reverse order, so each one is used
        #    at the level with the same resolution
        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = upsample(x)                    # double H and W, halve the channels
            x = torch.cat([skip, x], dim=1)    # skip connection: stack along the channel axis
            x = decoder(x)                     # mix "what" and "where"

        # 4. One score per class and per pixel, then remove the padding added at the beginning
        logits = self.head(x)
        return logits[..., :height, :width]


class UpBlock(nn.Module):
    """One decoder step: enlarge the deep map, concatenate the skip connection, mix with a DoubleConv.

    Unlike the transposed convolution of the hand-written U-Net, the deep map is enlarged with a
    bilinear interpolation to the exact size of the skip. This works for any input size (the
    ResNet halves the size rounding up, so sizes are not always exact multiples of 2).
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class ResNetUNet(nn.Module):
    """U-Net whose encoder is a ResNet (18 or 50), optionally pre-trained on ImageNet.

    The decoder is the same idea as in ``UNet`` (enlarge, concatenate the skip, DoubleConv), but
    the encoder is the ready-made ResNet of torchvision. With ``pretrained=True`` its weights
    come from ImageNet classification, so it already detects edges, textures and shapes before
    seeing any of our images (only the decoder is trained from scratch). With ``pretrained=False``
    every weight is random: same network, only the initialisation changes.

    Resolutions and channels for an input of H x W (the numbers in brackets). The channels of the
    skip connections are those of ResNet18 and, between square brackets, those of ResNet50, whose
    residual blocks are the deeper "bottleneck" ones::

        input                                   3 ch          (H, W)    ------------------------+ skip 0
        [stem]  conv 7x7 stride 2 + BN + ReLU  64 ch [64]     (H/2)    ---------------+ skip 1   |
        [pool]  max-pool 3x3 stride 2                         (H/4)                   |          |
        [layer1] residual blocks               64 ch [256]    (H/4)    ------+ skip 2  |          |
        [layer2] residual blocks              128 ch [512]    (H/8)    --+ skip 3     |          |
        [layer3] residual blocks              256 ch [1024]   (H/16)   + skip 4  |    |          |
        [layer4] residual blocks              512 ch [2048]   (H/32)     |       |    |          |
          | up4: enlarge, concat layer3, DoubleConv -> 256 <-------------+       |    |          |
          | up3: enlarge, concat layer2, DoubleConv -> 128 <---------------------+    |          |
          | up2: enlarge, concat layer1, DoubleConv -> 64  <--------------------------+          |
          | up1: enlarge, concat stem,   DoubleConv -> 32  <------------------------------------+
          | up0: enlarge, concat input (3 ch), DoubleConv -> 16                          (H, W)
        [head]  conv 1x1 -> num_classes

    The last level concatenates the (normalised) input image itself, so the decoder can use the
    exact pixel values for sharp boundaries and for small objects such as puddles.

    Args:
        num_classes: number of output channels.
        pretrained: start from the ImageNet weights of the ResNet (downloaded the first time).
        decoder_channels: output channels of up4, up3, up2, up1, up0.
        backbone: the encoder, a key of ``RESNET_BACKBONES``. ``ResNet50UNet`` is the same class
            with ``backbone="resnet50"``, so that it can be selected by name in the experiments.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True,
                 decoder_channels: tuple[int, int, int, int, int] = (256, 128, 64, 32, 16),
                 backbone: str = "resnet18"):
        super().__init__()
        if backbone not in RESNET_BACKBONES:
            raise ValueError(f"unknown backbone '{backbone}', available: {sorted(RESNET_BACKBONES)}")
        self.backbone_name = backbone
        spec = RESNET_BACKBONES[backbone]
        weights = spec["weights"] if pretrained else None
        backbone = getattr(torchvision.models, backbone)(weights=weights)
        c0, c1, c2, c3, c4 = spec["channels"]                      # channels of the five skip connections

        # Encoder: the pieces of the ResNet, without its final classification layers
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Decoder: UpBlock(channels of the deep map, channels of the skip, output channels)
        d4, d3, d2, d1, d0 = decoder_channels
        self.up4 = UpBlock(c4, c3, d4)
        self.up3 = UpBlock(d4, c2, d3)
        self.up2 = UpBlock(d3, c1, d2)
        self.up1 = UpBlock(d2, c0, d1)
        self.up0 = UpBlock(d1, 3, d0)
        self.head = nn.Conv2d(d0, num_classes, kernel_size=1)

        # Only the decoder is initialised here: the encoder keeps the pre-trained weights (or
        # the default initialisation of torchvision when it is trained from scratch)
        for module in [self.up4, self.up3, self.up2, self.up1, self.up0]:
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
        self.head.reset_parameters()

    def encoder_parameters(self):
        """Parameters of the ResNet part (to give them their own learning rate)."""
        for module in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            yield from module.parameters()

    def decoder_parameters(self):
        """Parameters of the decoder and of the output layer."""
        for module in (self.up4, self.up3, self.up2, self.up1, self.up0, self.head):
            yield from module.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder: every stage keeps its output as a skip connection
        x0 = self.stem(x)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # Decoder: back to the resolution of the input, one skip connection per level
        y = self.up4(x4, x3)
        y = self.up3(y, x2)
        y = self.up2(y, x1)
        y = self.up1(y, x0)
        y = self.up0(y, x)
        return self.head(y)


class ResNet50UNet(ResNetUNet):
    """``ResNetUNet`` with a ResNet50 encoder (23.5 M parameters instead of 11.2 M).

    The architecture is the one of ``ResNetUNet``: only the encoder changes. Its stages have more
    channels (256, 512, 1024, 2048 instead of 64, 128, 256, 512), so the decoder blocks that read
    the skip connections are wider at the input. Its pre-trained weights are the ImageNet ones
    of torchvision (version 2, trained with a more modern recipe than the original ones).
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True,
                 decoder_channels: tuple[int, int, int, int, int] = (256, 128, 64, 32, 16)):
        super().__init__(num_classes, pretrained, decoder_channels, backbone="resnet50")


MODELS = {
    "unet": UNet,
    "resnet18_unet": ResNetUNet,
    "resnet50_unet": ResNet50UNet,
}


def build_model(name: str = config.MODEL_NAME, **kwargs) -> nn.Module:
    """Create a model by name. Extra keyword arguments are passed to its constructor."""
    if name not in MODELS:
        raise ValueError(f"unknown model '{name}', available: {sorted(MODELS)}")
    return MODELS[name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def trace_shapes(model: UNet, input_shape: tuple[int, int, int, int] = (1, 3, 272, 512)):
    """Run a dummy forward pass and record the input and output shape of every main stage.

    Useful to see the architecture at work: how the resolution shrinks along the encoder,
    how the channels double, and how the concatenation with the skips doubles the input
    channels of every decoder block.

    Returns:
        A list of (stage name, input shape, output shape) in execution order.
    """
    stages = [(f"encoder {i + 1}", m) for i, m in enumerate(model.encoders)]
    stages.append(("pool", model.pool))
    stages.append(("bottleneck", model.bottleneck))
    stages += [(f"upsample {model.depth - i}", m) for i, m in enumerate(model.upsamples)]
    stages += [(f"decoder {model.depth - i}", m) for i, m in enumerate(model.decoders)]
    stages.append(("head", model.head))

    records = []

    def make_hook(name):
        def hook(module, inputs, output):
            records.append((name, tuple(inputs[0].shape), tuple(output.shape)))
        return hook

    handles = [module.register_forward_hook(make_hook(name)) for name, module in stages]
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        model(torch.zeros(input_shape, device=device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return records


def print_architecture(model: UNet, input_shape: tuple[int, int, int, int] = (1, 3, 272, 512)) -> None:
    """Print the stages of the model with their shapes, and the number of parameters."""
    print(f"{'stage':<14}{'input shape':<26}{'output shape':<26}")
    for name, shape_in, shape_out in trace_shapes(model, input_shape):
        print(f"{name:<14}{str(shape_in):<26}{str(shape_out):<26}")
    print(f"\ntrainable parameters: {count_parameters(model):,}")