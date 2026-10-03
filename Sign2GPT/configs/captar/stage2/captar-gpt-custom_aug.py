from ml_collections import config_dict
from pathlib import Path
import os
import numpy as np
import torch
from configs.base.base_utils import get_checkpoint_path, get_lmdb_path
import importlib
from train_utils.checkpoint_helpers import (
    get_best_checkpoint_details,
)


def get_config(train=True):
    cfg = config_dict.ConfigDict()
    cfg.name = Path(os.path.realpath(__file__)).stem
    base_name = Path(os.path.realpath(__file__)).parent.name
    code_path = str(Path(os.path.realpath(__file__)).resolve().parents[3])

    ckpt_path = get_checkpoint_path(base_name, cfg.name)
    lmdb_path = get_lmdb_path()
    if train:
        cfg.save_dir = get_save_dir(ckpt_path)
    else:
        cfg.save_dir = ckpt_path

    cfg.main_runner = "trainer.complete_translation_trainer"
    cfg.project_name = "captar_sign2gpt_s2"
    cfg.aug_name = "augmentation.video.custom_video_aug"

    # captar-libras_10-04_TORCH_MEAN                  = 0.15154793, 0.32615158, 0.20753308
    # captar-libras_10-04_TORCH_STD                   = 0.1986669,  0.17499135, 0.12384652

    cfg.aug_params = {
        "mean": [0.15154793, 0.32615158, 0.20753308],
        "std": [0.1986669, 0.17499135, 0.12384652],
        "strength": 0.2,
        "random_shift": 4,
        "stride": 2,
        "max_seq_len": 256,
        "cam_sim_kwargs": {
            "percentile_equalization": {"percentile_range": (0.80, 1)},
            "brightness": {"brightness_factor": (0.2, 2.5)},
            "contrast": {"contrast_factor": (0.2, 2)},
            "gaussian_noise": {"mean": 0, "std_dev": 5},
            "sharpness": {"sharpness_factor": (0.3, 1.8)},
            "perspective_transform": {"max_shift": 20}
        }
    }

    cfg.bs = int(
        2 * torch.cuda.device_count()
        # *        ((torch.cuda.mem_get_info()[1] / 10**6) / 24000)
    )
    cfg.accum = 1
    cfg.num_workers = min(min(cfg.bs, int(10 * torch.cuda.device_count())), 10)

    cfg.gate_grad_multiplier = 1.0

    base_bs = 8
    cfg.lr = 3e-4 * (cfg.bs**0.5) / (base_bs**0.5)

    # cfg.lr_scheduler = "customreduceonplateau"
    # cfg.lr_scheduler_params = config_dict.ConfigDict(
    #    {
    #        "name": "valid/avg_loss_ce",  # ou outro nome descritivo conforme sua implementação
    #        "mode": "min",  # Mode for ReduceLROnPlateau (e.g., "min" or "max")
    #        "factor": 0.1,  # Factor by which to reduce the learning rate
    #        "patience": 10,  # Number of epochs with no improvement before reducing LR
    #        "threshold_mode": "rel",  # "rel" or "abs", how the threshold is defined
    #        "threshold": 0.0001,  # Threshold for detecting no improvement
    #        "cooldown": 0,  # Number of epochs to wait before resuming normal operation
    #        "verbose": True,  # Print messages when reducing LR
    #        "metric_name": "valid/avg_loss_ce",  # Metric to track
    #    }
    # )

    cfg.lr_scheduler = "warmupwithcosine"
    cfg.lr_scheduler_params = config_dict.ConfigDict(
        {
            "lr_scale_factor": 0.01,
            "num_cycles": 1,
            "start_value_mult": 0.7,
            "end_value_mult": 0.7,
            "warmup_epochs": 15,
        }
    )

    cfg.optimizer_name = "adamw"
    cfg.optimizer_params = config_dict.ConfigDict(
        {
            "lr": cfg.lr,
            "weight_decay": 0.001,
            "betas": (0.9, 0.99),
        }
    )

    cfg.criterion_name = "losses.base_loss"

    cfg.criterion_params = config_dict.ConfigDict(
        {
            "dict_of_loss_params": {
                "ce": {
                    "cls_name": "losses.loss_functions.ce_loss",
                    "loss_params": {
                        "src_name": "logits",
                        "tgt_name": "gt_ids",
                        "mask_name": "gt_text_mask",
                        "label_smoothing": 0.1,
                    },
                    "weight": 1.0,
                },
            }
        }
    )

    cfg.max_epochs = 50
    cfg.model_checkpoint_dir = ""

    cfg.train_ds_name = "dataloaders.captar_video_dataset"
    cfg.valid_ds_name = "dataloaders.captar_video_dataset"
    cfg.test_ds_name = "dataloaders.captar_video_dataset"
    train_ds_params = {
        "split": "train",
        "pseudo_gloss_dir": f"{code_path}/data/captar-libras_10-04-gpt/processed_words.captar_pkl",
        "tsv_dir": f"{code_path}/data/captar-libras_10-04-gpt/sentence_label.tsv",
        "ds_params": {
            "lmdb_video_dir": f"{lmdb_path}/train",
            "isValid": False,
        },
        "shuffle": True,
        "num_workers": cfg.num_workers,
        "bs": cfg.bs,
        "drop_last": True,
    }
    cfg.train_ds_params = config_dict.ConfigDict(train_ds_params)

    valid_ds_params = {
        "split": "dev",
        "pseudo_gloss_dir": f"{code_path}/data/captar-libras_10-04-gpt/processed_words.captar_pkl",
        "tsv_dir": f"{code_path}/data/captar-libras_10-04-gpt/sentence_label.tsv",
        "ds_params": {
            "lmdb_video_dir": f"{lmdb_path}/dev",
            "isValid": True,
        },
        "shuffle": False,
        "num_workers": cfg.num_workers,
        "bs": cfg.bs,
        "drop_last": False,
    }
    cfg.valid_ds_params = config_dict.ConfigDict(valid_ds_params)

    test_ds_params = {
        "split": "test",
        "pseudo_gloss_dir": f"{code_path}/data/captar-libras_10-04-gpt/processed_words.captar_pkl",
        "tsv_dir": f"{code_path}/data/captar-libras_10-04-gpt/sentence_label.tsv",
        "ds_params": {
            "lmdb_video_dir": f"{lmdb_path}/test",
            "isValid": True,
        },
        "shuffle": False,
        "num_workers": cfg.num_workers,
        "bs": cfg.bs,
        "drop_last": False,
    }
    cfg.test_ds_params = config_dict.ConfigDict(test_ds_params)

    cfg.lm_name = "facebook/xglm-1.7B"
    cfg.additional_tokens = {
        # "pad_token": "<pad>",
        # "bos_token": ">",
        # "eos_token": ".",
    }
    cfg.pretext = ""

    # Not sure if this is needed for captar, but it is used in the original code
    # cfg.replacement_pickle = f"{code_path}/data/csldaily/csldaily_replacements.csl_pkl"

    cfg.stage1_name = "configs.captar.stage1.captar-gpt-custom_aug"
    mod = importlib.import_module(cfg.stage1_name, package=None)
    stage1_config = mod.get_config(train=False)

    stage1_name = stage1_config["model_name"]

    stage1_params = stage1_config["model_params"].to_dict()
    stage1_ckpt_dir = stage1_config["save_dir"]
    stage1_ckpt = get_best_checkpoint_details(
        stage1_ckpt_dir, best_checkpoint_name="_result_checkpoint_"
    )[0]
    assert stage1_ckpt is not None and stage1_ckpt != ""
    cfg.model_name = "models.trial_models.test_stage2_model"
    cfg.model_params = {
        "stage1_name": stage1_name,
        "stage1_params": stage1_params,
        "stage1_ckpt": stage1_ckpt,
        "post_name": "models.post_models.linear_pos_head",
        "post_params": {"pos_type": "sine", "pre_pos": True},
        "llm_name": cfg.lm_name,
        "lang_backbone_name": "models.huggingface.modeling_xglm",
        "adaptor_params": {
            "adapt_layers": list(np.arange(0, 24, 1)),
            "lora_layers": list(np.arange(0, 24, 1)),
            "w_lora_ff": False,
            "lora_rank": 4,
            "lora_drop": 0.1,
            "gate_type": "clamp",
            "lora_a": 4.0,
            "adapt_tokens": False,
        },
        "freeze": False,
    }

    cfg.gen_params = {"max_length": 64, "temperature": 1.0, "num_beams": 4}

    cfg.seed = 1
    cfg.grad_clip_norm = 1.0
    cfg.grad_clip_value = 1.0
    cfg.mixup = False
    cfg.logger_name = ["wandb"]  # ["wandb", "text"]
    cfg.resume = True
    cfg.train_length = None
    cfg.val_length = None
    cfg.log_every = 100
    cfg.save_ckpt = True
    cfg.score_factor = 1
    cfg.score_name = "valid/ableu"
    cfg.bfloat16_only = False

    # Set to `True` to perform character BLEU for CSL-Daily, else every sentence would be treated as 1 "word"
    # for training on other languages `apply_metric_splitter` should be set to `False`.
    cfg.apply_metric_splitter = False
    cfg.append_string = ""
    cfg.watch_grad = True
    return cfg


def get_save_dir(ckpt_path):
    if not os.path.exists(ckpt_path):
        save_dir = ckpt_path
    else:
        for i in range(1, 1000):
            candidate_path = f"{ckpt_path}_{i}"
            if not os.path.exists(candidate_path):
                save_dir = candidate_path
                break
    return save_dir
