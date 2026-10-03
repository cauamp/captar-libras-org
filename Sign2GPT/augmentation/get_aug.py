import importlib

def get_aug(aug_name, aug_params={}):
    if aug_name is not None:
        print(f"Using augmentation: {aug_name}")
        mod = importlib.import_module(aug_name, package=None)
        return mod.Transformation(**aug_params)
    else:
        return None
