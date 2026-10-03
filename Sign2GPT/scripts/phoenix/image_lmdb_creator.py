from argparse import ArgumentParser
import shutil
from pathlib import Path
import cv2
from PIL import Image
from time import time
import io
import lmdb
import pickle
import glob


if __name__ == "__main__":
    parser = ArgumentParser(parents=[])

    parser.add_argument(
        "--split",
        type=str,
        default="10-04",
    )

    split = parser.parse_args().split

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./datasets_hdd/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/raw_data/fullFrame-256x256px",
    )

    parser.add_argument(
        "--lmdb_dir",
        type=str,
        default="./datasets_hdd/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/lmdb_videos",
    )

    params, unknown = parser.parse_known_args()

    data_dir = Path(params.data_dir)

    video_paths = glob.glob(f"{data_dir}/*/*.mp4")

    print(f"Found {len(video_paths)} videos in {data_dir}")
    for video_path in video_paths:
        print(f"\n\nProcessing {video_path}")
        id = (
            video_path.split("/")[-2] + "/" +
            video_path.split("/")[-1].split(".mp4")[0]
        )

        lmdb_dir = Path(params.lmdb_dir) / id

        print(f"Creating lmdb for {id} \n \t at {lmdb_dir}")

        n_bytes = 2**40

        tmp_dir = Path("/tmp") / f"TEMP_{time()}"
        env = lmdb.open(path=str(tmp_dir), map_size=n_bytes)
        txn = env.begin(write=True)

        cap = cv2.VideoCapture(video_path)

        if lmdb_dir.exists() and lmdb_dir.is_dir():
            print(f"LMDB already exists for {id}, skipping...")
            continue

        lmdb_dir.mkdir(parents=True, exist_ok=True)

        ind = 0
        counter = 0
        while True:
            ret = cap.grab()
            if not ret:
                break
            ret, frame = cap.retrieve()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame).resize((256, 256))
            temp = io.BytesIO()
            img.save(temp, format="jpeg")
            temp.seek(0)
            txn.put(
                key=f"{ind}".encode("ascii"),
                value=temp.read(),
                dupdata=False,
            )
            ind += 1
            counter += 1

            if counter % 123 == 0 and counter != 0:
                txn.commit()
                txn = env.begin(write=True)

        txn.put(
            key="details".encode("ascii"),
            value=pickle.dumps({"num_frames": ind, "id": id}, protocol=4),
            dupdata=False,
        )
        txn.commit()

        env.close()

        if lmdb_dir.exists():
            shutil.rmtree(lmdb_dir)
        shutil.move(f"{tmp_dir}", f"{lmdb_dir}")
