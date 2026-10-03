from ml_collections import config_dict
import importlib

from train_utils.checkpoint_helpers import (
    get_best_checkpoint_details,
)


def get_config():
    cfg = config_dict.ConfigDict()

    cfg.main_runner = "trainer.complete_translation_trainer"
    cfg.aug_name = "augmentation.video.base_video_aug"

    # captar-libras_10-04_TORCH_MEAN                  = 0.15154793, 0.32615158, 0.20753308
    # captar-libras_10-04_TORCH_STD                   = 0.1986669,  0.17499135, 0.12384652

    cfg.aug_params = {
        "mean": [0.15154793, 0.32615158, 0.20753308],
        "std": [0.1986669, 0.17499135, 0.12384652],
        "strength": 0.2,
        "random_shift": 4,
        "stride": 2,
        "max_seq_len": 256,
    }

    cfg.stage2_name = "configs.captar.stage2.captar_example-2"

    cfg.gt_path = "data/captar-libras_10-04/sentence_label.tsv"
    cfg.videos_path = "./datasets_srv/captar-libras_10-04/raw_data/test/*.mp4"

    mod = importlib.import_module(cfg.stage2_name, package=None)
    stage2_config = mod.get_config(train=False)

    cfg.model_name = stage2_config.model_name

    stage2_ckpt_dir = stage2_config["save_dir"]
    stage2_ckpt = get_best_checkpoint_details(
        stage2_ckpt_dir, best_checkpoint_name="_result_checkpoint_"
    )[0]

    cfg.lm_name = stage2_config.lm_name
    cfg.model_params = stage2_config.model_params
    cfg.additional_tokens = stage2_config["additional_tokens"]
    assert stage2_ckpt is not None and stage2_ckpt != ""

    # cfg.checkpoint_path = stage2_ckpt
    cfg.checkpoint_path = 'work_dir/stage2/captar_example-2_1/best_result_checkpoint_6_95.7431.pt'
    cfg.gen_params = stage2_config.gen_params
    cfg.mixup = False

    return cfg
