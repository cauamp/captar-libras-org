import cv2
import glob
import os
import sys
import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('./preprocess/config.env')

# Retrieve environment variables
SRV_PATH = os.getenv('SRV_PATH')
DATASET_PATH = os.getenv('DATASET_PATH')

"""
Esse código faz a extração dos frames dos vídeos contidos
na pasta `dataset_root`, fazendo um center crop de 1080x1080
na imagem e salvando na pasta `videos_root`.
"""

# Para cada vídeo, extrai os frames e faz o center crop
def process_video(video_path, dataset_root, i, tam):
    video = cv2.VideoCapture(video_path)

    video_name = video_path.split('/')[-1]
    video_id = video_name[:-4]

    # Cria a pasta do vídeo
    video_root = os.path.join(dataset_root, video_id)
    if not os.path.isdir(video_root):
        os.mkdir(video_root)

    # Salva os frames
    ret, frame = video.read()
    id = 0
    while ret:
        crop = frame[:, 420:1500, :]
        cv2.imwrite(os.path.join(video_root, f'frame_{id:05d}.png'), crop)
        ret, frame = video.read()
        id += 1

    video.release()
    print(f'  {video_path} salvo ({i}/{tam}).')

def main():
    # Pega os argumentos
    parser = argparse.ArgumentParser('Extração de frames e center crop')
    parser.add_argument('--input', '-i', help='Pasta onde estão os vídeos que vão ser processados')
    parser.add_argument('--split', '-s', help='Pedaço do dataset que está sendo processado (train/dev/test)')
    args = parser.parse_args()

    # Checa se o split é válido
    if args.split not in ('train', 'dev', 'test'):
        print('O split do dataset deve ser um desses: [train, dev, test]')
        sys.exit()

    # Cria a pasta onde os frames serão salvos
    videos_root = os.path.join(args.input, args.split)

    # Lista os vídeos
    videos_list = glob.glob(os.path.join(videos_root, '*.mpeg'))
    num_videos = len(videos_list)
    print(f'\nProcessando {num_videos} vídeos.')

    # Cria a pasta onde os frames serão salvos
    dataset_root = os.path.join(DATASET_PATH, args.split)
    print(f'Frames serão salvos em {dataset_root}')
    if not os.path.isdir(dataset_root):
        os.mkdir(dataset_root)

    # Paraleliza a extração de frames
    with ThreadPoolExecutor(max_workers=60) as executor:
        i = 0
        futures = []
        for video_path in videos_list:
            futures.append(executor.submit(process_video, video_path, dataset_root, i, len(videos_list)))
            i += 1

        # Espera a conclusão de todas as tarefas
        for future in futures:
            future.result()

if __name__ == "__main__":
    main()
