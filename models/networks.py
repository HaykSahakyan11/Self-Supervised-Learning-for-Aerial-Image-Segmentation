import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.vision_transformer as vits
import utils.utils as dino_utils

from mmseg.models import BACKBONES, SEGMENTORS
from mmseg.models.segmentors import EncoderDecoder

from config import CONFIG, set_seed

set_seed(seed=42)
config = CONFIG()


@BACKBONES.register_module()
class DinoDeiTBackbone(nn.Module):
    def __init__(
            self, arch: str = "vit_small",
            patch_size: int = 8,
            pretrained_ckpt=None,
            out_indices=(3, 5, 7, 11),
            output: str = "pyramid",  # or "flat"
            num_classes: int = 8,
            embed_dim: int = 384,
            num_heads: int = 6,
            interpolate_mode='bicubic',
    ):
        super().__init__()
        self.vit = vits.__dict__[arch](
            patch_size=patch_size, num_classes=num_classes,
        )  # embed_dim and num_heads are set in the vits
        if pretrained_ckpt:
            dino_utils.load_pretrained_weights(
                model=self.vit,
                pretrained_weights=pretrained_ckpt,
                checkpoint_key=None,
                model_name=arch,
                patch_size=patch_size,
            )
        self.patch_size = patch_size
        self.out_indices = out_indices
        self.output = output

    def forward(self, x):
        B, _, H, W = x.shape

        # -- prepare tokens: patch embed + cls + pos --
        # vision_transformer.prepare_tokens does exactly that
        tokens = self.vit.prepare_tokens(x)  # (B, 1+N, C_embed)

        feats = []
        # -- run through transformer blocks, collect at out_indices --
        for i, blk in enumerate(self.vit.blocks):
            tokens = blk(tokens)
            if i in self.out_indices:
                # strip off CLS, reshape to (B, C, Hf, Wf)
                patch_tokens = tokens[:, 1:, :]  # (B, N, C)
                N, C_embed = patch_tokens.shape[1], patch_tokens.shape[2]
                Hf = H // self.patch_size
                Wf = W // self.patch_size
                feat = patch_tokens.transpose(1, 2).view(B, C_embed, Hf, Wf)
                feats.append(feat)

        # now feats = [feat3, feat5, feat7, feat11], each (B,384, Hf, Wf)
        if self.output == "flat":
            return tuple(feats)

        # -- build a {56,28,14,7} pyramid from the constant‐resolution feats --
        Hf = H // self.patch_size
        sizes = [2 * Hf, Hf, Hf // 2, Hf // 4]
        pyramid = []
        for feat, sz in zip(feats, sizes):
            if feat.shape[-2] != sz:
                feat = F.interpolate(feat, size=(sz, sz), mode="bilinear", align_corners=False)
            pyramid.append(feat)

        return tuple(pyramid)


@SEGMENTORS.register_module()
class UPerNetDinoDeiT(EncoderDecoder):
    def __init__(
            self, num_classes=8, backbone_type='DinoDeiTSmall',
            img_size=224, patch_size=8, feature_stack='pyramid',
            use_neck=False, pretrained_ckpt=None,
    ):
        weights_dir = config.model_weights_path
        if backbone_type == 'DinoDeiTSmall':
            embed_dim = config.vit_configs['vit_small']['embed_dim']
            num_heads = config.vit_configs['vit_small']['num_heads']
            out_indices = config.vit_configs['vit_small']['out_indices']
            out_layers_count = len(out_indices)  # 4
            if not pretrained_ckpt:
                if patch_size == 8:
                    pretrained_ckpt = config.dino_deit["vit_small"]["8"]
                elif patch_size == 16:
                    pretrained_ckpt = config.dino_deit["vit_small"]["16"]
                else:
                    raise NotImplementedError(f"Unsupported patch size: {patch_size}")
        elif backbone_type == 'DinoDeiTBase':
            embed_dim = config.vit_configs['vit_base']['embed_dim']
            num_heads = config.vit_configs['vit_base']['num_heads']
            out_indices = config.vit_configs['vit_base']['out_indices']
            out_layers_count = len(out_indices)  # 4
            if not pretrained_ckpt:
                if patch_size == 8:
                    pretrained_ckpt = config.dino_deit["vit_base"]["8"]
                elif patch_size == 16:
                    pretrained_ckpt = config.dino_deit["vit_base"]["16"]
                else:
                    raise NotImplementedError(f"Unsupported patch size: {patch_size}")

        else:
            raise NotImplementedError(f"Unsupported backbone: {backbone_type}")
        pretrained_ckpt = os.path.join(weights_dir, pretrained_ckpt)

        self.backbone_type = backbone_type
        backbone_cfg = dict(
            type="DinoDeiTBackbone",
            arch='vit_small',
            patch_size=patch_size,
            pretrained_ckpt=pretrained_ckpt,
            out_indices=out_indices,
            output=feature_stack,
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        decode_head_cfg = dict(
            type='UPerHead',
            in_channels=[embed_dim] * out_layers_count,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=False
        )

        auxiliary_head_cfg = dict(
            type='FCNHead',
            in_channels=embed_dim,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=False
        )

        if use_neck:
            neck_cfg = dict(
                type='FPN',
                in_channels=[embed_dim] * out_layers_count,
                out_channels=embed_dim,  # can be higher
                num_outs=out_layers_count
            )
        else:
            neck_cfg = None

        super().__init__(
            backbone=backbone_cfg,
            neck=neck_cfg,
            decode_head=decode_head_cfg,
            auxiliary_head=auxiliary_head_cfg,
            train_cfg=dict(),
            test_cfg=dict(mode='whole')
        )


@BACKBONES.register_module()
class DinoMCBackbone(nn.Module):
    """
    Wrap DINO‐MC’s ViT teacher (no classifier head) into an mmseg‐style backbone
    that returns a 4‐level pyramid of feature maps at strides {4,8,16,32}.
    """

    def __init__(
            self, arch: str = "vit_small",
            patch_size: int = 8,
            pretrained_ckpt=None,
            out_indices=(3, 5, 7, 11),
            output: str = "pyramid",  # or "flat"
            num_classes: int = 8,
    ):
        super().__init__()
        self.vit = vits.__dict__[arch](
            patch_size=patch_size, num_classes=num_classes,
        )
        if pretrained_ckpt:
            dino_utils.load_pretrained_weights(
                model=self.vit,
                pretrained_weights=pretrained_ckpt,
                checkpoint_key="teacher",
                model_name=arch,
                patch_size=patch_size,
            )
        self.patch_size = patch_size
        self.out_indices = out_indices
        self.output = output

    def forward(self, x):
        B, _, H, W = x.shape

        # -- prepare tokens: patch embed + cls + pos --
        # vision_transformer.prepare_tokens does exactly that
        tokens = self.vit.prepare_tokens(x)  # (B, 1+N, C_embed)

        feats = []
        # -- run through transformer blocks, collect at out_indices --
        for i, blk in enumerate(self.vit.blocks):
            tokens = blk(tokens)
            if i in self.out_indices:
                # strip off CLS, reshape to (B, C, Hf, Wf)
                patch_tokens = tokens[:, 1:, :]  # (B, N, C)
                N, C_embed = patch_tokens.shape[1], patch_tokens.shape[2]
                Hf = H // self.patch_size
                Wf = W // self.patch_size
                feat = patch_tokens.transpose(1, 2).view(B, C_embed, Hf, Wf)
                feats.append(feat)

        # now feats = [feat3, feat5, feat7, feat11], each (B,384, Hf, Wf)
        if self.output == "flat":
            return tuple(feats)

        # -- build a {56,28,14,7} pyramid from the constant‐resolution feats --
        Hf = H // self.patch_size
        sizes = [2 * Hf, Hf, Hf // 2, Hf // 4]
        pyramid = []
        for feat, sz in zip(feats, sizes):
            if feat.shape[-2] != sz:
                feat = F.interpolate(feat, size=(sz, sz), mode="bilinear", align_corners=False)
            pyramid.append(feat)

        return tuple(pyramid)


@SEGMENTORS.register_module()
class UPerNetDinoMC(EncoderDecoder):

    def __init__(
            self, num_classes: int = 8,
            backbone_type: str = "vit_small",
            pretrained_ckpt=None,
            img_size: int = 224,
            patch_size: int = 8,
            feature_stack='pyramid',
            use_neck=False
    ):
        self.backbone_type = backbone_type
        if self.backbone_type == 'vit_small':
            embed_dim = config.vit_configs['vit_small']['embed_dim']  # 384
            out_indices = config.vit_configs['vit_small']['out_indices']  # (3, 5, 7, 11)
            out_layers_count = len(out_indices)  # 4
        else:
            raise NotImplementedError(f"Unsupported backbone: {self.backbone_type}")

        self.use_neck = use_neck

        backbone_cfg = dict(
            type="DinoMCBackbone",
            arch=self.backbone_type,
            patch_size=patch_size,
            pretrained_ckpt=pretrained_ckpt,
            out_indices=out_indices,
            output=feature_stack,
        )

        decode_head_cfg = dict(
            type="UPerHead",
            in_channels=[embed_dim] * out_layers_count,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=dict(type="BN", requires_grad=True),
            align_corners=False,
        )

        if self.use_neck:
            neck_cfg = dict(
                type='FPN',
                in_channels=[embed_dim] * out_layers_count,
                out_channels=embed_dim,
                num_outs=out_layers_count
            )
        else:
            neck_cfg = None

        auxiliary_head_cfg = dict(
            type="FCNHead",
            in_channels=embed_dim,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=dict(type="BN", requires_grad=True),
            align_corners=False,
        )

        super().__init__(
            backbone=backbone_cfg,
            neck=neck_cfg,
            decode_head=decode_head_cfg,
            auxiliary_head=auxiliary_head_cfg,
            train_cfg=dict(),
            test_cfg=dict(mode="whole"),
        )


if __name__ == '__main__':
    def test_UPerNetDinoMC():
        # 1) Backbone feature shapes
        B, C, H, W = 2, 3, 224, 224
        x = torch.randn(B, C, H, W)

        # ckpt = config.dino_mc["vit_small"]["8"]
        ckpt = config.dino_deit["vit_small"]["8"]
        model_weights_path = config.model_weights_path
        ckpt_path = os.path.join(model_weights_path, ckpt)

        backbone = DinoMCBackbone(
            pretrained_ckpt=ckpt_path,
            output='pyramid'
        )
        feats = backbone(x)
        print("Backbone pyramid shapes:")
        for lvl, f in enumerate(feats):
            print(f"  level {lvl}: {tuple(f.shape)}")
        # → (2,384,56,56), (2,384,28,28), (2,384,14,14), (2,384,7,7)

        # 2) Full UPerNet output shape
        segmentor = UPerNetDinoMC(num_classes=8).eval()
        segmentor.init_weights()
        img_meta = [{
            'img_shape': (H, W, 3),
            'ori_shape': (H, W, 3),
            'pad_shape': (H, W, 3),
            'scale_factor': 1.0,
        } for _ in range(B)]
        logits = segmentor.encode_decode(x, img_meta)
        print("UPerNet output:", tuple(logits.shape))
        # → (2, 8, 224, 224)


    def test_UPerNetDinoDeiT():
        B, C, H, W = 2, 3, 224, 224
        x = torch.randn(B, C, H, W)

        ckpt = config.dino_deit["vit_small"]["8"]
        model_weights_path = config.model_weights_path
        ckpt_path = os.path.join(model_weights_path, ckpt)

        # we pass init_cfg so that init_weights() will load the checkpoint
        backbone = DinoDeiTBackbone(
            img_size=H,
            patch_size=8,
            in_channels=3,
            embed_dim=384,
            num_layers=12,
            num_heads=6,
            mlp_ratio=4,
            out_indices=(3, 5, 7, 11),
            interpolate_mode='bicubic',
            init_cfg=dict(type='Pretrained', checkpoint=ckpt_path)
        )
        backbone.init_weights()

        # forward: VisionTransformer.forward returns the [CLS] token embedding
        feats = backbone(x)  # shape: (B, embed_dim)
        print("Backbone shapes:")
        for lvl, f in enumerate(feats):
            print(f"  level {lvl}: {tuple(f.shape)}")
        # → (2, 384)

        # 3) test UPerNetDinoVit segmentor
        seg = UPerNetDinoDeiT(
            num_classes=8,
            backbone_type='DinoDeiTSmall',
            backbone_checkpoint=ckpt_path,
            img_size=H,
            patch_size=8
        )
        seg.init_weights()

        # build dummy img_meta for mmseg
        img_meta = [{
            'img_shape': (H, W, 3),
            'ori_shape': (H, W, 3),
            'pad_shape': (H, W, 3),
            'scale_factor': 1.0,
        } for _ in range(B)]

        # encode+decode → full-resolution logits
        logits = seg.encode_decode(x, img_meta)
        print(f"UPerNetDinoVit output shape: {tuple(logits.shape)}")
        # → (2, 8, 224, 224)


    # test_UPerNetDinoDeiT()
    test_UPerNetDinoMC()
