import os
import torch
import random
import numpy as np
import sklearn.utils


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)  # Python random module
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)  # NumPy
    sklearn.utils.check_random_state(seed)  # scikit-learn
    torch.manual_seed(seed)  # PyTorch
    torch.cuda.manual_seed(seed)  # For CUDA-based computations
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = True


BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
device = "cuda" if torch.cuda.is_available() else "cpu"


class CONFIG:
    def __init__(self):
        self.base_path = BASE_PATH
        self.dataset_path = os.path.join(BASE_PATH, 'datasets')
        self.model_weights_path = os.path.join(BASE_PATH, 'model_weights')
        self.upernet_dino_mc_path = os.path.join(self.model_weights_path, 'upernet_dino_mc')
        self.upernet_dino_deit_path = os.path.join(self.model_weights_path, 'upernet_dino_deit')
        self.inference_data_path = os.path.join(self.dataset_path, 'inference_data')

        self.dino_deit = {
            'vit_small': {
                '8': 'dino_deit/dino_deitsmall8_pretrain.pth',
                '16': 'dino_deit/dino_deitsmall16_pretrain.pth',
            },
            'vit_base': {
                '8': 'dino_deit/dino_vitbase8_pretrain',
                '16': 'dino_deit/dino_vitbase16_pretrain',
            }
        }

        self.dino_mc = {
            'vit_small': {
                '8': 'dino_mc/vit_mc_checkpoint300.pth',
            }
        }

        self.best_model_weights = {
            'upernet_dinomc': {
                'vit_small': {
                    'uavid': {
                        'patch_0': os.path.join(
                            self.upernet_dino_mc_path, 'uavid',
                            'best_checkpoint_dinomc_small_upernet_uavid_patch_0.pth'
                        ),
                        'patch_4': os.path.join(
                            self.upernet_dino_mc_path, 'uavid',
                            'best_checkpoint_dinomc_small_upernet_uavid_patch_4.pth'
                        ),
                        'patch_9': os.path.join(
                            self.upernet_dino_mc_path, 'uavid',
                            'best_checkpoint_dinomc_small_upernet_uavid_patch_9.pth'
                        ),
                        'patch_256': os.path.join(
                            self.upernet_dino_mc_path, 'uavid',
                            'c'
                        ),
                        'patch_512': os.path.join(
                            self.upernet_dino_mc_path, 'uavid',
                            'a'
                        ),
                        'patch_0_afine': os.path.join(
                            self.upernet_dino_mc_path, 'uavid',
                            'best_checkpoint_dinomc_small_upernet_uavid_patch_0_afine.pth'
                        ),
                    },
                },
            },
            'upernet_dinodeit': {
                'vit_small': {
                    'uavid': {
                        'patch_0': os.path.join(
                            self.upernet_dino_deit_path, 'uavid',
                            'best_checkpoint_dinodeitsmall_upernet_uavid_patch_0.pth' # 50.49
                        ),
                        'patch_4': os.path.join(
                            self.upernet_dino_deit_path, 'uavid',
                            'best_checkpoint_dinodeitsmall_upernet_uavid_patch_4.pth' # 57.46
                        ),
                        'patch_9': os.path.join(
                            self.upernet_dino_deit_path, 'uavid',
                            'b'
                        ),
                        'patch_256': os.path.join(
                            self.upernet_dino_deit_path, 'uavid',
                            'c'
                        ),	
                        'patch_512': os.path.join(
                            self.upernet_dino_deit_path, 'uavid',
                            'a'
                        ),
                        'patch_4_afine': os.path.join(
                            self.upernet_dino_deit_path, 'uavid',
                            'best_checkpoint_dinodeitsmall_upernet_uavid_patch_4_afine.pth' # 58.92
                        ),
                    }

                },
                'vit_base': {
                    'uavid': {
                        'patch_0': os.path.join(
                            self.upernet_dino_deit_path, 'uavid',
                            'a'
                        ),
                        'patch_4': os.path.join(
                            self.upernet_dino_deit_path, 'uavid',
                            'a'
                        ),
                        'patch_9': os.path.join(
                            self.upernet_dino_deit_path, 'uavid',
                            'b'
                        ),
                        'patch_256': os.path.join(
                            self.upernet_dino_deit_path, 'uavid',
                            'c'
                        ),
                        'patch_512': os.path.join(
                            self.upernet_dino_deit_path, 'uavid',
                            'd'
                        ),
                    }
                }
            }
        }

        self.best_models = {
            'UAVID': os.path.join(self.model_weights_path,
                                  'upernet/best_checkpoint_dinomcvitsmall_uavid_2_with_transformation.pth'),
            'POTSDAM': os.path.join(self.model_weights_path, 'upernet/best_checkpoint_dinomcvitsmall_potsdam_2.pth'),
            'LOVEDA': os.path.join(self.model_weights_path, 'upernet/best_checkpoint_dinomcvitsmall_loveda_2.pth'),
            'UAVID_patched': {
                'dino_deit': {
                    '4': os.path.join(
                        self.model_weights_path,
                        # 'upernet_patched/best_checkpoint_dinomcvitsmall_uavid_upernet_patched_2.pth'
                        'upernet_patched/best_checkpoint_dinodeitsmall_upernet_uavid_patch_4_afine.pth'
                    ),  # mIoU 50.0626
                },
                'dino_mc': {
                    '4': os.path.join(
                        self.model_weights_path,
                        'upernet_patched/best_checkpoint_dinomcvitsmall_uavid_upernet_new_patched_4_no_overlap.pth'
                    ),  # mIoU 52.6396
                }
            },
            'UDD6_patched': {
                'dino_deit': {
                    '0': os.path.join(
                        self.model_weights_path,
                        'upernet_dino_deit/udd6/best_checkpoint_dinodeit_small_upernet_udd6_patch_0_afine.pth'
                    ),  # mIoU 83.2541
                    '4': os.path.join(
                        self.model_weights_path,
                        # 'upernet_dino_deit/udd6/best_checkpoint_dinodeit_small_upernet_udd6_patch_4_afine.pth'
                        'upernet_dino_deit/udd6/best_checkpoint_dinodeitsmall_udd6_upernet_patched_4_dino_deit_afine_test_1.pth'
                    ),  # mIoU 86.5483
                },
                'dino_mc': {
                    '4': os.path.join(
                        self.model_weights_path,
                        'upernet_dino_mc/udd6/.pth'
                    ),  # mIoU 86.5483
                }
            },
        }

        self.vit_configs = {
            'vit_small': {
                'ckpt_key': 'vit_small',
                'embed_dim': 384,
                'num_layers': 12,
                'num_heads': 6,
                'mlp_ratio': 4,
                'drop_path_rate': 0.1,
                'out_indices': (3, 5, 7, 11)
            },
            'vit_base': {
                'ckpt_key': 'vit_base',
                'embed_dim': 768,
                'num_layers': 12,
                'num_heads': 12,
                'mlp_ratio': 4,
                'drop_path_rate': 0.1,
                'out_indices': (3, 5, 7, 11)
            }
        }

        self.UAVID = {
            "train": os.path.join(self.dataset_path, 'UAVID/train'),
            "val": os.path.join(self.dataset_path, 'UAVID/val'),
            "test": os.path.join(self.dataset_path, 'UAVID/test')
        }

        self.UAVID_patched = {
            '4': {
                "train": os.path.join(self.dataset_path, 'UAVID_patched_4/train'),
                "val": os.path.join(self.dataset_path, 'UAVID_patched_4/val'),
            },
            '9': {
                "train": os.path.join(self.dataset_path, 'UAVID_patched_9/train'),
                "val": os.path.join(self.dataset_path, 'UAVID_patched_9/val'),
            },
            '3_4': {
                "train": os.path.join(self.dataset_path, 'UAVID_patched_720_960_count_12/train'),
                "val": os.path.join(self.dataset_path, 'UAVID_patched_720_960_count_12/val'),
            },
            '360_384': {
                "train": os.path.join(self.dataset_path, 'UAVID_patched_360_384_count_60/train'),
                "val": os.path.join(self.dataset_path, 'UAVID_patched_360_384_count_60/val'),
            }
        }

        self.UAVID_patch_inf = {
            'dino_deit': {
                '4': {
                    "train": os.path.join(self.inference_data_path, 'deit', 'UAVID_patched_4/train'),
                    "val": os.path.join(self.inference_data_path, 'deit', 'UAVID_patched_4/val'),
                },
                '9': {
                    "train": os.path.join(self.inference_data_path, 'deit', 'UAVID_patched_9/train'),
                    "val": os.path.join(self.inference_data_path, 'deit', 'UAVID_patched_9/val'),
                },
                '224_224': {
                    "train": os.path.join(self.inference_data_path,
                                          'deit', 'UAVID_patched_240_240_count_144/train'),
                    "val": os.path.join(self.inference_data_path,
                                        'deit', 'UAVID_patched_240_240_count_144/val'),
                },
                '360_384': {
                    "train": os.path.join(self.inference_data_path, 'deit',
                                          'UAVID_patched_360_384_count_60/train'),
                    "val": os.path.join(self.inference_data_path, 'deit',
                                        'UAVID_patched_360_384_count_60/val'),
                }
            },
            'dino_mc': {
                '4': {
                    "train": os.path.join(self.inference_data_path, 'dino_mc', 'UAVID_patched_4/train'),
                    "val": os.path.join(self.inference_data_path, 'dino_mc', 'UAVID_patched_4/val'),
                },
                '9': {
                    "train": os.path.join(self.inference_data_path, 'dino_mc', 'UAVID_patched_9/train'),
                    "val": os.path.join(self.inference_data_path, 'dino_mc', 'UAVID_patched_9/val'),
                },
                '224_224': {
                    "train": os.path.join(self.inference_data_path,
                                          'dino_mc', 'UAVID_patched_240_240_count_144/train'),
                    "val": os.path.join(self.inference_data_path, 'dino_mc',
                                        'UAVID_patched_240_240_count_144/val'),
                },
                '360_384': {
                    "train": os.path.join(self.inference_data_path, 'dino_mc',
                                          'UAVID_patched_360_384_count_60/train'),
                    "val": os.path.join(self.inference_data_path, 'dino_mc',
                                        'UAVID_patched_360_384_count_60/val'),
                }
            },
        }

        self.UDD6 = {
            "train": os.path.join(self.dataset_path, 'UDD6/train'),
            "val": os.path.join(self.dataset_path, 'UDD6/val'),
            "metadata": os.path.join(self.dataset_path, 'UDD6/metadata')
        }

        self.UDD6_patched = {
            '4': {
                "train": os.path.join(self.dataset_path, 'UDD6_patched_4/train'),
                "val": os.path.join(self.dataset_path, 'UDD6_patched_4/val'),
            },
            '9': {
                "train": os.path.join(self.dataset_path, 'UDD6_patched_9/train'),
                "val": os.path.join(self.dataset_path, 'UDD6_patched_9/val'),
            },
        }

        self.UDD6_patch_inf = {
            'dino_deit': {
                '0': {
                    'train': os.path.join(self.inference_data_path, 'deit', 'UDD6_patched_0/train'),
                    'val': os.path.join(self.inference_data_path, 'deit', 'UDD6_patched_0/val'),
                },
                '4': {
                    "train": os.path.join(self.inference_data_path, 'deit', 'UDD6_patched_4/train'),
                    "val": os.path.join(self.inference_data_path, 'deit', 'UDD6_patched_4/val'),
                },
            }
        }

        self.POTSDAM = {
            "train": os.path.join(self.dataset_path, 'POTSDAM'),
            "val": os.path.join(self.dataset_path, 'POTSDAM'),
            "test": os.path.join(self.dataset_path, 'POTSDAM')
        }

        self.LOVEDA = {
            "train": os.path.join(self.dataset_path, 'LOVEDA/Train'),
            "val": os.path.join(self.dataset_path, 'LOVEDA/Val'),
            "test": os.path.join(self.dataset_path, 'LOVEDA/Test')
        }

        self.train_configs = {
            'num_epochs': 100,
            # 'learning_rate': 1e-4,
            'learning_rate': 3e-4,
            'weight_decay': 1e-4,
            'eta_min': 1e-6,
            'warmup_epochs': 5,
            'warmup_lr': 1e-6,
            'lr_scheduler': 'cosine',
            'lr_decay_epochs': [30, 60],
            'lr_decay_rate': 0.1,
            'epochs': 100,
        }

        self.agg_train_configs = {
            'num_epochs': 50,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'eta_min': 1e-6,
            'base_channels': 64,
        }

        # self.batch_size = 2
        self.batch_size = 4
        self.image_size = 224
        self.big_image_size = 512
        self.patch_count = 4
        self.patch_size = 8

        self.wandb_api_key = ""
