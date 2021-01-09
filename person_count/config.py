import torch
import os
import os.path as osp

CURRENT_PATH = osp.dirname(osp.realpath(__file__))


class ModelConfig:
    device = (
        torch.device("cpu")
        if not torch.cuda.is_available() or os.getenv("FORCE_CPU", "0") == "1"
        else torch.device("cuda")
    )

    root_dir = osp.realpath(osp.join(CURRENT_PATH, ".."))
    data_dir = osp.join(root_dir, "data")


model_config = ModelConfig()
