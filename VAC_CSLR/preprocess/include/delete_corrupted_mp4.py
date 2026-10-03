import os
import pandas as pd
from dotenv import load_dotenv
import cv2

def remove_corrupted_videos():
    # Carrega variáveis de ambiente do arquivo config.env
    load_dotenv('./config.env')

    # Definir o caminho raiz do dataset
    DATASET_ROOT = os.getenv('DATASET_ROOT')

    # Carregar o arquivo CSV com erros
    errors_df = pd.read_csv('./errors.csv')

    # Filtrar os video_ids com Erro igual a 'video_cap' e 'WAS CORRUPTED'
    video_ids = pd.concat([
        errors_df[errors_df['Erro'] == 'WAS CORRUPTED']['VideoID'],
        errors_df[errors_df['Erro'] == 'VIDEO NOT OPENED']['VideoID'], 
        errors_df[errors_df['Erro'] == 'NONE IN CROP']['VideoID'], 
    ])

    # Remover os vídeos nas pastas de 'train', 'dev', 'test'
    for video_id in video_ids:
        removed = False
        for split in ['train', 'dev', 'test']:
            video_folder_path = os.path.join(DATASET_ROOT, 'raw_data', split)
            output_path = os.path.join(video_folder_path, f'{video_id}.mp4')

            # Verificar se o arquivo existe antes de tentar abrir
            if os.path.exists(output_path):
                video_cap = None
                try:
                    # Tenta abrir o vídeo para verificar se está corrompido
                    video_cap = cv2.VideoCapture(output_path)
                    if not video_cap.isOpened() or video_cap.read()[0] is False:
                        os.remove(output_path)
                        print(f"Removed {output_path}")
                        removed = True
                        break
                    else:
                        print(f'{video_id} está íntegro.')
                        break
                finally:
                    # Libera o vídeo caso ele tenha sido aberto
                    if video_cap is not None:
                        video_cap.release()
        if not removed:
            print(f'{video_id} não foi encontrado')
            
if __name__ == '__main__':
    remove_corrupted_videos()