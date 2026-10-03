import cv2
import os
from argparse import ArgumentParser
from pathlib import Path
import glob


def arguments():
    parser = ArgumentParser(parents=[])

    parser.add_argument(
        "--image_dir",
        type=str,
        default="datasets_hdd/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/{}/",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="datasets_hdd/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/raw_data/fullFrame-256x256px/{}/",
    )

    params, unknown = parser.parse_known_args()
    return params


def images_to_video(split_dirs, save_dirs, fps=25):
    for image_dir, save_dir in zip(split_dirs, save_dirs):
        split_folders = os.listdir(image_dir)

        for image_folder in split_folders:
            video_name = image_folder.split("/")[-1]
            video_folder = image_dir + video_name
            tmp_dir = "/tmp"

            Path(tmp_dir).mkdir(parents=True, exist_ok=True)
            image_files = [
                f for f in glob.glob(f'{video_folder}/*.png')
            ]
            image_files.sort()

            first_image = cv2.imread(image_files[0])
            height, width, channels = first_image.shape

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v")  # Use 'XVID' for AVI format
            out = cv2.VideoWriter(f"{tmp_dir}/{video_name}.mp4",
                                  fourcc, fps, (256, 256))
            if not out.isOpened():
                raise RuntimeError(
                    f"Falha ao abrir VideoWriter para {video_name}.mp4 – verifique codec e fps.")

            for image_path in image_files:
                og_img = cv2.imread(image_path)
                if og_img is None:
                    raise Exception("Erro: imagem original está vazia.")
                # Step 1: Stretch vertically from 210x260 to 210x300
                stretched_img = cv2.resize(
                    og_img, (210, 300), interpolation=cv2.INTER_LINEAR)

                # Step 2: Resize to 256x256
                final_img = cv2.resize(
                    stretched_img, (256, 256), interpolation=cv2.INTER_LINEAR)

                out.write(final_img)

            out.release()

            cp_cmd = "cp {} {}"
            # copy video to tmp
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            os.system(
                cp_cmd.format(f"{tmp_dir}/{video_name}.mp4",
                              f"{save_dir}{video_name}.mp4")
            )


if __name__ == "__main__":

    args = arguments()
    image_dir = [args.image_dir.format(x) for x in ["train", "dev", "test"]]
    output_dir = [args.save_dir.format(x) for x in ["train", "dev", "test"]]
    images_to_video(image_dir, output_dir)
