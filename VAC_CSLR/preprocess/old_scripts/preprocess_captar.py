import os
import csv
from concurrent.futures import ThreadPoolExecutor
from load_annotations import load_annotations
from dotenv import load_dotenv

# Load parameters from .env file
load_dotenv('./preprocess/config.env')

TEST_CSV_PATH = os.getenv('TEST_CSV_PATH')
VALIDATE_CSV_PATH = os.getenv('VALIDATE_CSV_PATH')
TRAIN_CSV_PATH = os.getenv('TRAIN_CSV_PATH')
DATASET_ROOT = os.getenv('DATASET_ROOT')
NUM_WORKERS = int(os.getenv('NUM_WORKERS'))

def process_video_cut(video_id, video, split, i, annotations):
    os.system(f'python3 time_crop.py --video {video} --videoID {video_id} --folder {DATASET_ROOT}/{split} --annotations {annotations}')
    print(f'  {video} salvo ({i}/{len(videos[split])}).')

videos = {}

with open(TEST_CSV_PATH, 'r') as file:
    reader = csv.reader(file)
    next(reader)
    videos['test'] = []
    for row in reader:
        pair = (row[0], row[1])
        videos['test'].append(pair)

with open(VALIDATE_CSV_PATH, 'r') as file:
    reader = csv.reader(file)
    next(reader)
    videos['dev'] = []
    for row in reader:
        pair = (row[0], row[1])
        videos['dev'].append(pair)

with open(TRAIN_CSV_PATH, 'r') as file:
    reader = csv.reader(file)
    next(reader)
    videos['train'] = []
    for row in reader:
        pair = (row[0], row[1])
        videos['train'].append(pair)

train_size = len(videos['train'])
dev_size = len(videos['dev'])
test_size = len(videos['test'])
total_size = train_size + dev_size + test_size

print(f'Dataset size: {total_size}')
print(f'  Train: {train_size}')
print(f'  Dev: {dev_size}')
print(f'  Test: {test_size}')

# Load annotations
annotations = load_annotations()

# Process and save new videos
print(f'\nCortando e salvando os novos vídeos...')
for split in ['train', 'dev', 'test']:
    print(f'\nSplit: {split}')
    # Create directories if they don't exist
    dataset_root = f'{DATASET_ROOT}/raw_data/{split}'
    if not os.path.isdir(dataset_root):
        os.makedirs(dataset_root)

    # Parallelize the cutting process
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        i = 0
        futures = []
        for video in videos[split]:
            future = executor.submit(process_video_cut, video[0], video[1], split, i, annotations)
            futures.append(future)
            i += 1

        # Wait for all tasks to complete
        for future in futures:
            future.result()

    print(f'\nSplit {split}: salvos em {dataset_root}')

# extrai os frames e faz o center crop
print('\nFazendo a extração de frames e center crop...')
for split in ['train','dev','test']:
    print(f'\nSplit: {split}')
    os.system(f'python3 frame_center_crop.py --split {split} --input {DATASET_ROOT}/raw_data')
    
print('\nTodos os processamentos foram finalizados.')
