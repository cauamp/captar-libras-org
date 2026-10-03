from absl import app
import importlib
import torch
import ignite.distributed as idist
import os
from ml_collections import config_flags
import warnings
import logging

os.environ["TRITON_CACHE_DIR"] = "/tmp"
os.environ["TRANSFORMERS_CACHE"] = "/tmp"
torch.hub.set_dir("/tmp")

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress Albumentations update check
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# Set the logging level to WARNING to suppress lower levels like INFO and DEBUG
logging.basicConfig(level=logging.WARNING)

# Example for controlling specific library loggers
logging.getLogger('ignite').setLevel(logging.WARNING)
logging.getLogger('onnxscript').setLevel(logging.WARNING)


def main(_):
    mod = importlib.import_module(CONFIG.value.main_runner, package=None)
    backend = "nccl"
    nproc_per_node = (
        torch.cuda.device_count() if torch.cuda.device_count() > 1 else None
    )

    with idist.Parallel(
        backend,
        # master_port=random.randint(49152, 65535)
        nproc_per_node=nproc_per_node,
    ) as parallel:
        parallel.run(mod.Trainer, CONFIG.value)


if __name__ == "__main__":
    CONFIG = config_flags.DEFINE_config_file(
        "config",
        default="",
    )

    try:
        import torch.multiprocessing as mp
        # 'force=True' override any previous settings.
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    app.run(main)
