import os
import glob
import pandas as pd
import gspread
import re
from oauth2client.service_account import ServiceAccountCredentials
from load_annotations import load_sentences
from dotenv import load_dotenv

# Load parameters from .env file
load_dotenv('./preprocess/config.env')

# Retrieve environment variables
FRAME_FOLDER = os.getenv('FRAME_FOLDER')
ANNOTATION_ROOT = os.getenv('ANNOTATION_ROOT')

# Define o caminho dos folders de frames
folder = FRAME_FOLDER  # Now using the environment variable

# Lê o arquivo de anotações
print('Lendo a planilha de sentenças...')
sentences_path = load_sentences(version='new')
sentences = pd.read_csv(sentences_path)
print('Planilha de sentenças lida.\n')

for split in ['train', 'dev', 'test']:
    # Lê os nomes das pastas
    input_folder = os.path.join(folder, split, '*/')
    videos_list = sorted(glob.glob(input_folder))
    
    print(f'Criando o arquivo de anotações para o split {split} ({len(videos_list)} vídeos)...')
    
    # Cria o arquivo de anotações
    annotation_root = ANNOTATION_ROOT  # Now using the environment variable
    if not os.path.isdir(annotation_root):
        os.makedirs(annotation_root)
    
    annotation_name = os.path.join(annotation_root, split + '.csv')
    with open(annotation_name, 'a') as annotation_file:
        annotation_file.write('id|folder|signer|annotation\n')
    
        # Para cada pasta, busca as glosas na planilha
        for video in videos_list:
            # Pega o prefixo do arquivo
            video_folder = video.split('/')[-2]
            video_name = video_folder
        
            # Pega o ID da sentença
            reg_exp_s = r"_s([^_\s]+)"
            s_id = re.search(reg_exp_s, video_name)
            sentence_id = s_id.group(1)
        
            # Pega o ID da pessoa sinalizando
       
