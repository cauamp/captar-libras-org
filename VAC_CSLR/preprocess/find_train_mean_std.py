import os
import cv2
from tqdm import tqdm
import glob
import numpy as np
import concurrent.futures

NUM_WORKERS = 64
dataset = "captar-libras_10-04"

inputs_list = np.load(
    f"./{dataset}/train_info.npy", allow_pickle=True
).item()


def read_videos(index_len):
    imgs = []
    for idx in range(0, index_len):
        fi = inputs_list[idx]
        img_folder = os.path.join(
            f"../datasets_srv/{dataset}/features/fullFrame-256x256px/"
            + fi["folder"]
        )

        img_list = sorted(glob.glob(img_folder))
        img_l = [img_path for img_path in img_list]
        # img_l = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)/255 for img_path in img_list]
        imgs.extend(img_l)
    return imgs


def process_batch(batch_files):
    imgs = [
        cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) / 255 for file in batch_files
    ]
    imgs = np.array(imgs)
    means = np.mean(imgs, axis=(0, 1, 2))
    stds = np.std(imgs, axis=(0, 1, 2))
    return means, stds


if __name__ == "__main__":
    num_videos = list(inputs_list.keys())[-1]
    imgs_l = read_videos(num_videos)
    batch_size = NUM_WORKERS

    data = {"means": [], "std": []}
    for i in tqdm(range(0, len(imgs_l), batch_size), desc="Processando batches"):
        if i + batch_size < len(imgs_l):
            batch_files = imgs_l[i : i + batch_size]
        else:
            batch_files = imgs_l[i:]

        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = []
            for j in range(0, batch_size, 2):
                batch = batch_files[j : j + 2]
                if len(batch) > 0:
                    futures.append(executor.submit(process_batch, batch))

            for future in concurrent.futures.as_completed(futures):
                means, stds = future.result()
                data["means"].append(means)
                data["std"].append(stds)

    print("data mean", len(data["means"]) * 2)
    print("data std ", len(data["std"]) * 2)

    overall_mean = np.mean(np.array(data["means"]), axis=0)
    overall_std = np.mean(np.array(data["std"]), axis=0)

    print(f"Overall mean: {overall_mean}")
    print(f"Overall std: {overall_std}")
    
    with open(f"{dataset}/{dataset}_train_mean_std.txt", "w") as f:
        f.write(f"Overall mean: {overall_mean}\n")
        f.write(f"Overall std: {overall_std}\n")
        f.close()
