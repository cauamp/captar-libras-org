import os
import csv
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import pandas as pd
import cv2
from include.histogram_equalization import equalizeImg
from include.video_detection import detection_batch
from include.delete_corrupted_mp4 import remove_corrupted_videos
from tqdm import tqdm
from PIL import Image
import numpy as np
import shutil
import subprocess
from enum import Enum

# Carrega variáveis de ambiente do arquivo .env
load_dotenv('./config.env')

# Carrega os caminhos dos CSVs e outras variáveis de ambiente
TEST_CSV_PATH = os.getenv('TEST_CSV_PATH')
VALIDATE_CSV_PATH = os.getenv('VALIDATE_CSV_PATH')
TRAIN_CSV_PATH = os.getenv('TRAIN_CSV_PATH')
CONTROL_CSV_PATH = os.getenv('CONTROL_CSV_PATH')
DATASET_ROOT = os.getenv('DATASET_ROOT')
NUM_WORKERS = int(os.getenv('NUM_WORKERS'))
ANNOTATIONS_PATH = os.getenv('ANNOTATIONS_PATH')
FRAME_FOLDER = os.getenv('FRAME_FOLDER')
OUTPUT_RES = os.getenv('OUTPUT_RES')
ANNOTATION_PREFIX = os.getenv('ANNOTATION_PREFIX')
EQUALIZE = bool(os.getenv('EQUALIZE'))

# Lista para armazenar erros
errors = []
# Define a tolerancia limite para pixels quase pretos
tolerance = 20 

class CropMode(Enum):
    CENTRAL = 0  # Crop central
    STATIC = 1   # YOLO Crop estático
    DYNAMIC = 2  # YOLO Crop dinâmico
    BOTH = 3     # YOLO Crop estático e dinâmico

CROP_MODE = CropMode(int(os.getenv('CROP_MODE')))
    
def get_time_annotations(annotations, video_id, datetime):
    """Fetches time annotations for the given datetime from the CSV."""
    try:
        time1 = annotations.loc[annotations['Datetime'] == datetime, 'time_1'].iat[0]
        time2 = annotations.loc[annotations['Datetime'] == datetime, 'time_2'].iat[0]
        if (not time1 and time1 != 0) or not time2:
            raise ValueError('Blank time columns')
    except Exception:
        #print(f"!! ERRO !!, anotação de tempo de {video_id} ({datetime}) não encontrada na planilha. Vídeo não processado.")
        errors.append([video_id, datetime, 'NOT FOUND IN CSV'])
        return None, None
    return float(time1 / 1000), float(time2 / 1000)

def check_existing_output(video_id, output_path, datetime):
    """Checks if the video has already been processed and is intact."""
    if os.path.exists(output_path):
        video_cap = cv2.VideoCapture(output_path)
        if not video_cap.isOpened() or not video_cap.read()[0]:
            print(f'  {video_id} já foi processado, mas está corrompido. Reprocessando...')
            os.remove(output_path)
            video_cap.release()

        else:
            video_cap.release()
            #print(f'  {video_id} já foi processado e salvo.')
            return True
    return False

def initialize_video_cap(video_path):
    """Initializes video capture object."""
    video_cap = cv2.VideoCapture(video_path.replace('.mp4', '.avi'))
    # Verifica se o vídeo foi aberto com sucesso
    if not video_cap.isOpened():
        # Define o caminho para salvar a conversão .mp4
        conversion_path = os.path.join('../datasets_srv/conversions', os.path.basename(video_path).replace('.mp4', '.avi'))
        final_path = conversion_path.replace('.avi', '.mp4')
        
        # Verifica se o .mp4 convertido já existe
        if os.path.exists(final_path):
            video_cap = cv2.VideoCapture(final_path)
        
        # Se ainda não conseguiu abrir, tenta conversão
        if not video_cap.isOpened():
            os.remove(final_path)
            
            # Verifica se o .avi original existe para conversão
            avi_source = video_path.replace('.mp4', '.avi')
            if os.path.exists(avi_source):
                print(f'    {avi_source} não pode ser aberto. Convertendo para .mp4...')
                
                # Garante que o diretório de conversões existe
                os.makedirs(os.path.dirname(conversion_path), exist_ok=True)
                
                # Copia o arquivo .avi para o diretório de conversões
                shutil.copy2(avi_source, conversion_path)

                # Converte o .avi para .mp4 usando ffmpeg
                cmd = f"ffmpeg -i {conversion_path} -c copy {final_path}"
                subprocess.run(cmd, shell=True, check=True)

                # Remove o arquivo .avi após a conversão
                os.remove(conversion_path)

                # Tenta abrir o novo arquivo convertido
                video_cap = cv2.VideoCapture(final_path)

    return video_cap

def initialize_video_writer(output_path, dsize):
    """Initializes video writer object."""
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, dsize)
    return video_writer

def process_frame(frame, crop_values, dsize, central_crop=False):
    """Processes a frame based on the crop mode and resizes it."""
    if central_crop:
        crop = frame[:, 420:1500, :]
        return cv2.resize(crop, dsize, interpolation=cv2.INTER_LANCZOS4)
    
    cropped = frame.crop(crop_values)
    resized_frame = cropped.resize(dsize, resample=Image.Resampling.LANCZOS)
    final_frame = np.array(resized_frame)
    return cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)

def adjust_bounding_box(x0, y0, x1, y1, max_width, max_height):
    """Adjusts the bounding box to ensure the correct aspect ratio."""
    bounding_box_height = y1 - y0
    bounding_box_width = x1 - x0

    if bounding_box_height < bounding_box_width:
        # Se a altura for menor que a largura, ajusta a altura
        missing_padding = bounding_box_width - bounding_box_height 
        y0 -= missing_padding / 2  # Ajusta y0 para cima
        y1 += missing_padding / 2  # Ajusta y1 para baixo
       
       # Garante que os limites da altura estão dentro dos limites do frame
        if y0 < 0:
            y0 = 0
        if y1 > max_height:
            y1 = max_height
            y0 = y1 - bounding_box_width  # Ajusta y0 para manter a proporção
    else:
        # Garante uma proporção correta, preenchendo a largura se a altura for maior
        missing_padding = bounding_box_height - bounding_box_width
        y1 -= missing_padding / 2  # Ajusta y1 para cima
        x0 -= missing_padding / 4  # Ajusta x0 para a esquerda
        x1 += missing_padding / 4  # Ajusta x1 para a direita

        # Garante que os limites da largura estão dentro dos limites do frame
        if x0 < 0:
            x0 = 0
        if x1 > max_width:
            x1 = max_width
            x0 = x1 - bounding_box_height  # Ajusta x0 para manter a proporção

    return int(x0), int(y0), int(x1), int(y1)

def process_video(video_path, video_id, datetime, annotations, dsize, video_folder_path, frames_folder_path, person_id, crop_mode=CROP_MODE):
    """Main function to process video frames, crop, resize, and save output."""
    errorReturn = video_id, None, None, None

    # Pega os termos de anotação do CSV
    time1, time2 = get_time_annotations(annotations, video_id, datetime)

    # Verifica se as colunas de tempo estão vazias
    if (not time1 and time1 != 0) or not time2:
        return errorReturn

    # Variáveis auxiliares
    fps = 30
    gap = 1  # Segundos de intervalo antes e depois do sinal para evitar cortes
    first_frame = max(int((time1 - gap) * fps), 0)
    end_frame = int((time2 + gap) * fps)
    annotation = annotations.loc[annotations['Datetime'] == datetime, 'Gloss'].iat[0]
    output_path = os.path.join(video_folder_path, f'{video_id}.mp4')

    frame_folder = os.path.join(frames_folder_path, f'{video_id}')
    if crop_mode in [CropMode.CENTRAL, CropMode.STATIC, CropMode.BOTH]:
        os.makedirs(frame_folder, exist_ok=True)
        os.makedirs(video_folder_path, exist_ok=True)

    if crop_mode in [CropMode.DYNAMIC, CropMode.BOTH]:
        dynamic_output = output_path.replace('raw_data', 'dynamic_raw_data')
        dynamic_frame_folder = frame_folder.replace('features', 'dynamic_features')
        os.makedirs(video_folder_path.replace('raw_data', 'dynamic_raw_data'), exist_ok=True)
        os.makedirs(dynamic_frame_folder, exist_ok=True)

    # Check if video is already processed based on crop_mode
    if crop_mode == CropMode.DYNAMIC:
        if check_existing_output(video_id, dynamic_output, datetime):
            return video_id, f'{video_id}/*.png', f'Signer{person_id:04d}', annotation

    elif crop_mode in [CropMode.STATIC, CropMode.CENTRAL]:
        if check_existing_output(video_id, output_path, datetime):
            return video_id, f'{video_id}/*.png', f'Signer{person_id:04d}', annotation

    elif crop_mode == CropMode.BOTH:
        if check_existing_output(video_id, output_path, datetime) and check_existing_output(video_id, dynamic_output, datetime):
            return video_id, f'{video_id}/*.png', f'Signer{person_id:04d}', annotation

    print(f'Processando: {video_id} ({datetime})')	
    video_cap = initialize_video_cap(video_path)
    if not video_cap.isOpened():
        print(f'    ERRO: Não foi possível abrir o vídeo {video_id}.')
        errors.append([video_id, datetime, 'VIDEO NOT OPENED'])
        return errorReturn
    
    if crop_mode == CropMode.CENTRAL:
        video_writer = initialize_video_writer(output_path, dsize)
        
    #print('  Processando:', video_id)
    # Crop-related settings
    max_box_height = 0
    frames_list = []
    success = True
    frame_count = 0
    saved_frame_count = 0

    # Processa os frames do vídeo
    while success:
        success, frame = video_cap.read()
        frame_count += 1

        # Verifica se o frame está dentro do intervalo especificado
        if success and (frame_count >= first_frame) and (frame_count <= end_frame):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame_count > 10 and np.all(frame <= tolerance):
                errors.append([video_id, datetime, 'IMPORTANT NEAR-BLACK FRAME'])
                return errorReturn
            if frame_count < 10 and np.all(frame <= tolerance):
                continue

            if EQUALIZE == 1:
                frame = equalizeImg(frame)  # Equalize the image to brighten it up

            if crop_mode == CropMode.CENTRAL:
                finalFrame = process_frame(frame, None, dsize, central_crop=True)
                frame_filename = os.path.join(frame_folder, f'frame_{saved_frame_count:05d}.png')
                cv2.imwrite(frame_filename, finalFrame)
                video_writer.write(finalFrame)

            else:
                frames_list.append(frame)

            saved_frame_count += 1

        elif frame_count > end_frame:
            break
    if video_cap.isOpened():
        video_cap.release()
    if crop_mode == CropMode.CENTRAL:
        video_writer.realease()
        
    if crop_mode != CropMode.CENTRAL: 
        video_writer = None           
        if crop_mode in [CropMode.DYNAMIC, CropMode.BOTH] :
            video_writer = initialize_video_writer(dynamic_output, dsize)
            
        boundarys_list = detection_batch(frames_list)
        frames_list = [Image.fromarray(frame) for frame in frames_list]

        for i, ((x0, y0, x1, y1), frame) in enumerate(zip(boundarys_list, frames_list)):
            # Find the bounding box for the person in the frame                
            if None in (x0, y0, x1, y1):
                errors.append([video_id, datetime, 'NONE IN CROP'])
                video_writer.release()
                return errorReturn

            frame_width, frame_height = frame.size
            crop_values = adjust_bounding_box(x0, y0, x1, y1, frame_width, frame_height)

            boundingBoxHeight = y1 - y0    
            if boundingBoxHeight > max_box_height:
                max_box_height = boundingBoxHeight
                max_box_crop_values = crop_values

            if crop_mode in [CropMode.DYNAMIC, CropMode.BOTH]:
                finalFrame = process_frame(frame, crop_values, dsize)
                video_writer.write(finalFrame)

                # Salva o frame processado na pasta de frames dinamicos
                frame_filename = os.path.join(dynamic_frame_folder, f'frame_{i:05d}.png')
                cv2.imwrite(frame_filename, finalFrame)


        # Libera os recursos de vídeo
        if video_writer is not None:
            video_writer.release()

        if crop_mode in [CropMode.STATIC, CropMode.BOTH]:
            video_writer = initialize_video_writer(output_path, dsize)
            for i, frame in enumerate(frames_list):
                finalFrame = process_frame(frame, max_box_crop_values, dsize)
                video_writer.write(finalFrame)
                # Salva o frame processado na pasta de frames
                frame_filename = os.path.join(frame_folder, f'frame_{i:05d}.png')
                cv2.imwrite(frame_filename, finalFrame)

            # Libera os recursos de vídeo
            video_writer.release()

    #print(f'  {video_id} processado e salvo.')
    return video_id, f'{video_id}/*.png', f'Signer{person_id:04d}', annotation

# Função principal
def main():
    print('Variaveis de ambiente:')
    for key in os.environ:
        print(f'{key}: {os.getenv(key)}')    
    
    print(
    f"Erro de compatibilidade: 'opencv-python==4.6.0.66' apresenta inconsistências na abertura de vídeos .avi.\n"
    f"Para resolver, utilize a versão 'opencv-python==4.10.0.84' ou superior.\n"
    f"Versão atual instalada: {cv2.__version__}"
    )

    # Carrega as anotações em um DataFrame
    annotations = pd.read_csv(ANNOTATIONS_PATH)
    
    # Define o tamanho de saída para os vídeos processados
    dsize = tuple(map(int, OUTPUT_RES[1:-1].split(',')))

    # Dicionário para armazenar os vídeos por split (treino, validação, teste)
    videos = {}

    # Carrega os vídeos a partir dos arquivos CSV
    csv_paths = [
        ('test', TEST_CSV_PATH),
        ('dev', VALIDATE_CSV_PATH),
        ('train', TRAIN_CSV_PATH),
    ]

    create_control = False
    if CONTROL_CSV_PATH:
        csv_paths.append(('ctrl', CONTROL_CSV_PATH))
        create_control = True
        
    
    for split_name, csv_path in csv_paths:
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Pula a linha de cabeçalho
            videos[split_name] = []
            for row in reader:
                quad = (row[-2], row[-1], row[0], int(row[2]))  # Lê id do vídeo, caminho, data/hora e person_id
                videos[split_name].append(quad)

    # Calcula o tamanho dos datasets
    train_size = len(videos['train'])
    dev_size = len(videos['dev'])
    test_size = len(videos['test'])
    total_size = train_size + dev_size + test_size


    # Exibe o tamanho dos datasets
    print(f'Dataset size: {total_size}')
    print(f'  Train: {train_size}')
    print(f'  Dev: {dev_size}')
    print(f'  Test: {test_size}')
    if create_control:
        control_size = len(videos['ctrl'])
    print(f'  Control: {control_size}')

    print('\nCortando e salvando os vídeos...')
    
    # Processa os vídeos para cada split (treino, validação, teste)
    splits = ['train', 'dev', 'test']
    if create_control:
        splits.append('ctrl')
        
    for split in splits:
        print(f'\nSplit: {split}')

        # Define os diretórios para salvar os vídeos e frames processados
        video_folder_path = f'{DATASET_ROOT}/raw_data/{split}'
        
        frame_folder_path = FRAME_FOLDER.replace('(dsize)', f'{dsize[0]}x{dsize[1]}')
        frame_folder_path = os.path.join(frame_folder_path, split)
        if not os.path.isdir(frame_folder_path):
            os.makedirs(frame_folder_path)
        
        split_data = []
        # Usa ThreadPoolExecutor para paralelizar o processo de corte dos vídeos
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = []
            for i, video in enumerate(videos[split]):
                video_id = video[0]
                video_path = video[1].replace('/srv/projects2/captarlibras_finep/dataset_captarlibras', '../raw_dataset_captarlibras')
                date = video[2]
                person_id = video[3]
                
                # Envia cada vídeo para processamento em paralelo
                future = executor.submit(process_video, video_path, video_id, date, annotations, dsize, video_folder_path, frame_folder_path, person_id)
                futures.append(future)
            # Aguarda o término de todos os vídeos enviados para processamento
            for future in tqdm(futures):
                result = future.result()
                if None not in result:
                    split_data.append(result)

        # Escreve o csv que servira de base para gloss_dict
        annotation_csv = f"{DATASET_ROOT}/{ANNOTATION_PREFIX.format(split)}"
        annotation_dir = os.path.dirname(annotation_csv)
        if not os.path.isdir(annotation_dir):
            os.makedirs(annotation_dir)
            
        with open(annotation_csv, mode='w', newline='') as file: 
            writer = csv.writer(file, delimiter='|') 
            writer.writerow(['id', 'folder', 'signer', 'annotation']) 
            for row in split_data:
                writer.writerow(row)
                
        print(f'\n{split}: Videos salvos em {video_folder_path} Anotações salvas em {annotation_csv}')
    
    print('\nTodos os processamentos foram finalizados.')
    
    # Salva os erros em um arquivo CSV
    error_file = './errors.csv'
    with open(error_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['VideoID', 'Datetime', 'Erro'])
        writer.writerows(errors)
    
    print('Removendo .mp4 corrompidos')
    remove_corrupted_videos()
    print(f'Arquivo de erros salvo em: {error_file}')

# Executa a função principal
if __name__ == "__main__":
    main()
