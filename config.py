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
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = True


BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))


class CONFIG:
    def __init__(self):
        self.base_path = BASE_PATH
        self.dataset_path = os.path.join(BASE_PATH, 'datasets')
        self.model_weights_path = os.path.join(BASE_PATH, 'model_weights')

        self.dino_mc_checkpoint = {
            'vit_small': {
                '8': 'dino_mc/dino_deitsmall8_pretrain.pth0',
                '16': 'dino_mc/dino_deitsmall16_pretrain.pth',
            },
            'vit_base': {
                '8': 'dino_mc/dino_vitbase8_pretrain',
                '16': 'dino_mc/dino_vitbase16_pretrain',
            }
        }

        self.UAVID = {
            "train": os.path.join(self.dataset_path, 'UAVID/train'),
            "val": os.path.join(self.dataset_path, 'UAVID/val'),
            "test": os.path.join(self.dataset_path, 'UAVID/test')
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

        self.batch_size = 4
        self.image_size = 224

        self.wandb_api_key = "c7bb6c8c1316577f4e864eb2de43c2165825172c"
