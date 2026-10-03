from pathlib import Path


def get_checkpoint_path(base_name, name):
    ckpt_path = Path("./work_dir") / base_name / name
    return ckpt_path


def get_lmdb_path():
    lmdb_path = "datasets_srv/captar-libras_2026/lmdb_videos"
    if not Path(lmdb_path).exists():
        raise FileNotFoundError(f"LMDB path {lmdb_path} does not exist.")
    if not Path(lmdb_path).is_dir():
        raise NotADirectoryError(f"LMDB path {lmdb_path} is not a directory.")
    return lmdb_path
