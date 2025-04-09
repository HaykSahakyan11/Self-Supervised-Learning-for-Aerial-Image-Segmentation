from mmengine.runner import load_checkpoint
from mmseg.models import BACKBONES, SEGMENTORS
from mmseg.models.backbones import VisionTransformer
from mmseg.models.segmentors import EncoderDecoder

from config import CONFIG, set_seed

set_seed(seed=42)
config = CONFIG()


@BACKBONES.register_module()
class DinoMCViT(VisionTransformer):
    """
    Custom mmseg backbone for DINO-MC ViT pretrained weights.
    """

    def __init__(
            self,
            img_size=224,
            patch_size=8,
            in_channels=3,
            embed_dims=384,
            num_layers=12,
            num_heads=6,
            mlp_ratio=4,
            out_indices=(3, 5, 7, 11),
            drop_path_rate=0.1,
            interpolate_mode='bicubic',
            init_cfg=None,
            **kwargs
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop_path_rate=drop_path_rate,
            with_cls_token=True,
            out_indices=out_indices,
            interpolate_mode=interpolate_mode,
            **kwargs
        )
        self.init_cfg = init_cfg

    def init_weights(self):
        if self.init_cfg and 'checkpoint' in self.init_cfg:
            load_checkpoint(self, self.init_cfg['checkpoint'], strict=False)
            print(f"[DinoMCViT] Loaded DINO-MC checkpoint from {self.init_cfg['checkpoint']}")
        else:
            super().init_weights()
            print("[DinoMCViT] No pretrained checkpoint provided.")


@SEGMENTORS.register_module()
class UPerNetDinoMCViT(EncoderDecoder):
    def __init__(
            self, num_classes=8, backbone_type='DinoMCViTSmall',
            backbone_checkpoint=None, img_size=224,
            patch_size=8
    ):
        if backbone_type == 'DinoMCViTSmall':
            embed_dims, num_heads = 384, 6
            out_indices = (3, 5, 7, 11)
        elif backbone_type == 'DinoMCViTBase':
            embed_dims, num_heads = 768, 12
            out_indices = (3, 5, 7, 11)
        else:
            raise NotImplementedError(f"Unsupported backbone: {backbone_type}")

        self.backbone_type = backbone_type
        backbone_cfg = dict(
            type='DinoMCViT',
            img_size=img_size,
            patch_size=patch_size,
            embed_dims=embed_dims,
            num_layers=12,
            num_heads=num_heads,
            out_indices=out_indices,
            interpolate_mode='bicubic',
            init_cfg=dict(type='Pretrained', checkpoint=backbone_checkpoint)
        )

        decode_head_cfg = dict(
            type='UPerHead',
            in_channels=[embed_dims] * 4,
            in_index=[0, 1, 2, 3],
            # TODO Larger values for pool_scales example: (1, 2, 8, 12)
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=False
        )

        auxiliary_head_cfg = dict(
            # TODO: ASPPHead
            type='FCNHead',
            in_channels=embed_dims,
            # TODO: in_index=1 or 3
            in_index=2,
            # TODO: channels=384
            channels=256,
            # TODO: num_convs=2
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=False
        )

        super().__init__(
            backbone=backbone_cfg,
            neck=None,
            decode_head=decode_head_cfg,
            auxiliary_head=auxiliary_head_cfg,
            train_cfg=dict(),
            test_cfg=dict(mode='whole')
        )