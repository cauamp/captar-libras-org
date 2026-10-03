from ml_collections import config_dict
import importlib

from train_utils.checkpoint_helpers import (
    get_best_checkpoint_details,
)


def get_config():
    cfg = config_dict.ConfigDict()

    cfg.main_runner = "trainer.complete_translation_trainer"
    cfg.aug_name = "augmentation.video.inference_aug"

    # captar-libras_10-04_TORCH_MEAN                  = 0.15154793, 0.32615158, 0.20753308
    # captar-libras_10-04_TORCH_STD                   = 0.1986669,  0.17499135, 0.12384652

    cfg.aug_params = {
        "mode":  "yolo_hf",
    }

    cfg.stage2_name = "configs.captar.stage2.captar-gpt-custom_aug"

    cfg.videos_path = "datasets_srv/testes_cam_brio/*.mp4"
    cfg.gt_path = 'datasets_srv/testes_cam_brio/gt.csv'
    # cfg.videos_path = "datasets_srv/testes_cam_zed/*.avi"
    # cfg.gt_path = 'datasets_srv/testes_cam_zed/gt.csv'

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
    cfg.checkpoint_path = 'work_dir/stage2/captar-gpt-custom_aug/best_result_checkpoint_8_93.7254.pt'
    cfg.gen_params = stage2_config.gen_params
    cfg.mixup = False

    return cfg
