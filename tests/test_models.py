"""Tests of src/models.py.

A model is hard to check against a known answer, so the tests verify its structure
(shapes, parameters), properties that must always hold (the output of an image does not
depend on the other images of the batch, gradients reach every weight) and that the
network is able to learn.
"""
import pytest
import torch
import torchvision
from torch import nn

from src.losses import SegmentationLoss, compute_class_weights
from src.models import (DoubleConv, ResNet50UNet, ResNetUNet, UNet, UpBlock, build_model, count_parameters,
                        trace_shapes)


@pytest.fixture(scope="module")
def small_model():
    torch.manual_seed(0)
    return UNet(base_channels=4, depth=3)


# --- Shapes ----------------------------------------------------------------------------------
@pytest.mark.parametrize("batch, size", [(1, (32, 48)), (3, (64, 64)), (2, (16, 80))])
def test_output_has_one_score_map_per_class_at_input_resolution(small_model, batch, size):
    output = small_model(torch.randn(batch, 3, *size))
    assert output.shape == (batch, 9, *size)
    assert output.dtype == torch.float32


@pytest.mark.parametrize("size", [(35, 50), (33, 33), (17, 100), (9, 9)])
def test_sizes_that_are_not_multiples_of_the_pooling_factor_are_supported(small_model, size):
    assert small_model(torch.randn(2, 3, *size)).shape == (2, 9, *size)


def test_number_of_classes_and_input_channels_are_configurable():
    model = UNet(num_classes=5, in_channels=1, base_channels=4, depth=2)
    assert model(torch.randn(1, 1, 16, 16)).shape == (1, 5, 16, 16)


def test_depth_must_be_positive():
    with pytest.raises(ValueError):
        UNet(depth=0)


# --- Structure -------------------------------------------------------------------------------
def test_double_conv_changes_only_the_channels():
    block = DoubleConv(3, 8)
    assert block(torch.randn(2, 3, 10, 12)).shape == (2, 8, 10, 12)
    assert sum(isinstance(m, nn.Conv2d) for m in block) == 2
    assert all(m.bias is None for m in block if isinstance(m, nn.Conv2d))   # BatchNorm makes it redundant


def test_channels_double_at_every_level():
    model = UNet(base_channels=8, depth=4)
    encoder_out = [encoder[-2].num_features for encoder in model.encoders]   # BatchNorm of the 2nd conv
    assert encoder_out == [8, 16, 32, 64]
    assert model.bottleneck[-2].num_features == 128


def test_parameter_count_of_the_default_configuration_is_about_two_million():
    # base_channels=16 and depth=4 are the defaults of config.py
    assert 1.5e6 < count_parameters(UNet(base_channels=16, depth=4)) < 2.5e6


def test_trace_shows_resolution_halving_and_skip_concatenation():
    model = UNet(base_channels=4, depth=3)
    records = trace_shapes(model, (1, 3, 32, 48))
    by_name = {}
    for name, shape_in, shape_out in records:
        by_name.setdefault(name, []).append((shape_in, shape_out))

    # Encoder: same resolution as the input at the first level, then halved at each level
    assert by_name["encoder 1"][0] == ((1, 3, 32, 48), (1, 4, 32, 48))
    assert by_name["encoder 2"][0][1] == (1, 8, 16, 24)
    assert by_name["encoder 3"][0][1] == (1, 16, 8, 12)
    assert by_name["bottleneck"][0][1] == (1, 32, 4, 6)
    assert len(by_name["pool"]) == 3
    # Decoder blocks receive the upsampled map concatenated with the skip: twice the channels
    for level in (1, 2, 3):
        (shape_in, shape_out), = by_name[f"decoder {level}"]
        assert shape_in[1] == 2 * shape_out[1]
    assert by_name["head"][0][1] == (1, 9, 32, 48)


def test_trace_restores_the_training_mode():
    model = UNet(base_channels=4, depth=2).train()
    trace_shapes(model, (1, 3, 16, 16))
    assert model.training


# --- Properties that must always hold ----------------------------------------------------------
def test_in_eval_mode_an_image_gives_the_same_output_alone_or_inside_a_batch(small_model):
    small_model.eval()
    batch = torch.randn(4, 3, 32, 48)
    with torch.no_grad():
        together = small_model(batch)
        alone = small_model(batch[2:3])
    assert torch.allclose(together[2:3], alone, atol=1e-5)


def test_every_decoder_block_receives_the_matching_encoder_output_as_skip():
    """The first half of the channels entering a decoder block must be, exactly, the output
    of the encoder block that works at the same resolution (the skip connection)."""
    torch.manual_seed(0)
    model = UNet(base_channels=4, depth=3).eval()
    encoder_outputs, decoder_inputs = {}, {}
    handles = []
    for level, encoder in enumerate(model.encoders):
        handles.append(encoder.register_forward_hook(
            lambda m, i, o, level=level: encoder_outputs.__setitem__(level, o)))
    for index, decoder in enumerate(model.decoders):
        handles.append(decoder.register_forward_hook(
            lambda m, i, o, index=index: decoder_inputs.__setitem__(index, i[0])))
    with torch.no_grad():
        model(torch.randn(1, 3, 32, 48))
    for handle in handles:
        handle.remove()

    for index in range(model.depth):
        level = model.depth - 1 - index                             # decoders go from deep to shallow
        skip = encoder_outputs[level]
        assert torch.equal(decoder_inputs[index][:, : skip.shape[1]], skip)


def test_every_parameter_receives_a_gradient():
    torch.manual_seed(0)
    model = UNet(base_channels=4, depth=2)
    model(torch.randn(2, 3, 16, 16)).sum().backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert missing == []


def test_initial_logits_are_not_extreme():
    """A good initialisation starts with small scores, i.e. a loss close to ln(9) = 2.2."""
    torch.manual_seed(0)
    model = UNet(base_channels=8, depth=3)
    logits = model(torch.randn(2, 3, 32, 32))
    assert logits.abs().max() < 20


def test_batch_norm_starts_as_the_identity():
    for module in UNet(base_channels=4, depth=2).modules():
        if isinstance(module, nn.BatchNorm2d):
            assert torch.all(module.weight == 1) and torch.all(module.bias == 0)


# --- Learning ----------------------------------------------------------------------------------
def test_the_network_can_overfit_a_small_batch():
    torch.manual_seed(0)
    images = torch.randn(4, 3, 32, 32)
    # Labels that depend on the position and on the colour: solvable by a convolutional net
    target = (images[:, 0] > 0).long() + 2 * (torch.arange(32).view(1, 1, 32) > 15).long()
    model = UNet(base_channels=8, depth=2)
    loss_fn = SegmentationLoss(compute_class_weights([1, 1, 1, 1, 1, 1, 1, 1, 1]))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    first = loss_fn(model(images), target).item()
    for _ in range(60):
        optimizer.zero_grad()
        loss = loss_fn(model(images), target)
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.3 * first


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_mixed_precision_forward_and_backward_on_gpu():
    model = UNet(base_channels=8, depth=3).cuda()
    images = torch.randn(2, 3, 64, 96, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(images)
    output.float().mean().backward()
    assert output.shape == (2, 9, 64, 96)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


# --- Factory -----------------------------------------------------------------------------------
def test_build_model_creates_a_unet_by_name():
    model = build_model("unet", base_channels=4, depth=2)
    assert isinstance(model, UNet)
    assert model.depth == 2


def test_build_model_rejects_unknown_names():
    with pytest.raises(ValueError):
        build_model("does_not_exist")
        

# --- ResNetUNet (experiments B and C) ----------------------------------------------------------
@pytest.fixture(scope="module")
def resnet_model():
    torch.manual_seed(0)
    return ResNetUNet(pretrained=False)


@pytest.mark.parametrize("batch, size", [(1, (64, 96)), (2, (35, 50)), (2, (272, 512))])
def test_resnet_unet_output_has_the_input_resolution(resnet_model, batch, size):
    resnet_model.eval()
    with torch.no_grad():
        assert resnet_model(torch.randn(batch, 3, *size)).shape == (batch, 9, *size)


def test_resnet_unet_every_parameter_receives_a_gradient():
    torch.manual_seed(0)
    model = ResNetUNet(pretrained=False)
    model(torch.randn(2, 3, 64, 96)).sum().backward()
    assert [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()] == []


def test_encoder_and_decoder_parameters_partition_the_model(resnet_model):
    encoder = {id(p) for p in resnet_model.encoder_parameters()}
    decoder = {id(p) for p in resnet_model.decoder_parameters()}
    everything = {id(p) for p in resnet_model.parameters()}
    assert encoder.isdisjoint(decoder)
    assert encoder | decoder == everything


def test_the_resnet18_encoder_has_the_expected_size(resnet_model):
    encoder_parameters = sum(p.numel() for p in resnet_model.encoder_parameters())
    assert 11.0e6 < encoder_parameters < 11.4e6                         # ResNet18 without its classifier


def test_up_block_works_with_sizes_that_are_not_exact_doubles():
    block = UpBlock(in_channels=8, skip_channels=4, out_channels=6)
    out = block(torch.randn(1, 8, 5, 9), torch.randn(1, 4, 9, 17))
    assert out.shape == (1, 6, 9, 17)


@pytest.mark.parametrize("pretrained, expected", [(True, "IMAGENET1K_V1"), (False, None)])
def test_pretrained_flag_selects_the_imagenet_weights(monkeypatch, pretrained, expected):
    """Checked without downloading anything: the torchvision constructor is replaced by a recorder."""
    requested = {}
    real_resnet18 = torchvision.models.resnet18

    def recorder(weights=None, **kwargs):
        requested["weights"] = weights
        return real_resnet18(weights=None)

    monkeypatch.setattr(torchvision.models, "resnet18", recorder)
    ResNetUNet(pretrained=pretrained)
    assert (requested["weights"].name if requested["weights"] is not None else None) == expected


def test_build_model_creates_the_resnet_unet_by_name():
    assert isinstance(build_model("resnet18_unet", pretrained=False), ResNetUNet)


# Channels of (stem, layer1, layer2, layer3, layer4) of the two supported encoders
ENCODERS = [(ResNetUNet, (64, 64, 128, 256, 512)), (ResNet50UNet, (64, 256, 512, 1024, 2048))]


@pytest.mark.parametrize("model_class, channels", ENCODERS)
def test_resnet_encoder_stages_have_the_expected_resolution_and_channels(model_class, channels):
    model = model_class(pretrained=False).eval()
    outputs = {}
    handles = [getattr(model, name).register_forward_hook(
        lambda m, i, o, name=name: outputs.__setitem__(name, o))
        for name in ("stem", "layer1", "layer2", "layer3", "layer4")]
    with torch.no_grad():
        model(torch.randn(1, 3, 64, 96))
    for handle in handles:
        handle.remove()

    factors = {"stem": 2, "layer1": 4, "layer2": 8, "layer3": 16, "layer4": 32}
    for (name, factor), expected_channels in zip(factors.items(), channels):
        assert outputs[name].shape[1:] == (expected_channels, 64 // factor, 96 // factor), name


@pytest.mark.parametrize("model_class", [ResNetUNet, ResNet50UNet])
def test_every_resnet_decoder_level_receives_the_matching_skip(model_class):
    """up4 must get layer3, up3 layer2, up2 layer1, up1 the stem and up0 the input image itself."""
    torch.manual_seed(0)
    model = model_class(pretrained=False).eval()
    produced, skips = {}, {}
    handles = [getattr(model, name).register_forward_hook(
        lambda m, i, o, name=name: produced.__setitem__(name, o))
        for name in ("stem", "layer1", "layer2", "layer3")]
    handles += [getattr(model, name).register_forward_hook(
        lambda m, i, o, name=name: skips.__setitem__(name, i[1]))
        for name in ("up4", "up3", "up2", "up1", "up0")]
    image = torch.randn(1, 3, 64, 96)
    with torch.no_grad():
        model(image)
    for handle in handles:
        handle.remove()

    assert torch.equal(skips["up4"], produced["layer3"])
    assert torch.equal(skips["up3"], produced["layer2"])
    assert torch.equal(skips["up2"], produced["layer1"])
    assert torch.equal(skips["up1"], produced["stem"])
    assert torch.equal(skips["up0"], image)


# --- ResNet50UNet (experiment E) ---------------------------------------------------------------
@pytest.fixture(scope="module")
def resnet50_model():
    torch.manual_seed(0)
    return ResNet50UNet(pretrained=False)


@pytest.mark.parametrize("batch, size", [(1, (64, 96)), (2, (35, 50))])
def test_resnet50_unet_output_has_the_input_resolution(resnet50_model, batch, size):
    resnet50_model.eval()
    with torch.no_grad():
        assert resnet50_model(torch.randn(batch, 3, *size)).shape == (batch, 9, *size)


def test_resnet50_unet_every_parameter_receives_a_gradient():
    torch.manual_seed(0)
    model = ResNet50UNet(pretrained=False)
    model(torch.randn(2, 3, 64, 96)).sum().backward()
    assert [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()] == []


def test_resnet50_encoder_and_decoder_parameters_partition_the_model(resnet50_model):
    encoder = {id(p) for p in resnet50_model.encoder_parameters()}
    decoder = {id(p) for p in resnet50_model.decoder_parameters()}
    assert encoder.isdisjoint(decoder)
    assert encoder | decoder == {id(p) for p in resnet50_model.parameters()}


def test_the_resnet50_encoder_has_the_expected_size(resnet50_model):
    encoder_parameters = sum(p.numel() for p in resnet50_model.encoder_parameters())
    assert 23.0e6 < encoder_parameters < 24.0e6                         # ResNet50 without its classifier


def test_resnet18_models_are_unchanged_by_the_support_for_other_backbones():
    """Experiments B, C and D were trained before ResNet50 was added: their checkpoints must still load,
    which requires the same layers, with the same names and shapes."""
    model = ResNetUNet(pretrained=False)
    assert model.backbone_name == "resnet18"
    assert count_parameters(model) == 14_328_649
    first_decoder_conv = model.up4.conv[0]
    assert first_decoder_conv.weight.shape == (256, 512 + 256, 3, 3)


def test_resnet50_decoder_reads_the_wider_skip_connections(resnet50_model):
    assert resnet50_model.up4.conv[0].weight.shape == (256, 2048 + 1024, 3, 3)
    assert resnet50_model.up1.conv[0].weight.shape == (32, 64 + 64, 3, 3)       # d2 + stem channels


@pytest.mark.parametrize("pretrained, expected", [(True, "IMAGENET1K_V2"), (False, None)])
def test_resnet50_pretrained_flag_selects_the_imagenet_weights(monkeypatch, pretrained, expected):
    """Checked without downloading anything: the torchvision constructor is replaced by a recorder."""
    requested = {}
    real_resnet50 = torchvision.models.resnet50

    def recorder(weights=None, **kwargs):
        requested["weights"] = weights
        return real_resnet50(weights=None)

    monkeypatch.setattr(torchvision.models, "resnet50", recorder)
    ResNet50UNet(pretrained=pretrained)
    assert (requested["weights"].name if requested["weights"] is not None else None) == expected


def test_build_model_creates_the_resnet50_unet_by_name():
    model = build_model("resnet50_unet", pretrained=False)
    assert isinstance(model, ResNet50UNet) and model.backbone_name == "resnet50"


def test_unknown_backbone_is_rejected():
    with pytest.raises(ValueError):
        ResNetUNet(pretrained=False, backbone="resnet1000")
