import cv2
import glob
import os
import pandas as pd
import argparse
import gspread
import sys
import re

# pega os argumentos
parser = argparse.ArgumentParser('Corte dos vídeos de acordo com a anotação fraca')
parser.add_argument('--video', '-v', help='Arquivo de vídeo a ser processado')
parser.add_argument('--videoID', '-vID', help='Arquivo de vídeo a ser processado')
parser.add_argument('--folder', '-f', help='Pasta onde salvar o arquivo de output')
parser.add_argument('--annotations', '-a', help='Caminho para o arquivo de anotações')
args = parser.parse_args()

# pega o nome do vídeo
video_name = args.video.split('/')[-1]

video_prefix = args.videoID

# lê o arquivo de anotações
annot = pd.read_csv(args.annotations)

# pega as anotações de tempo da planilha
try:
    time1 = annot.loc[annot['video_id']==video_prefix,'time_1'].iat[0]
    time2 = annot.loc[annot['video_id']==video_prefix,'time_2'].iat[0]

except Exception as e:
    print(f"!! ERRO !!, anotacao de tempo do video_id {video_prefix} não encontrada na planilha. Vídeo não processado.")
    sys.exit()    

# checa se o tempo está vazio
if time1 == '':
    print(f'    ERRO: A coluna time_1 para o vídeo {video_name} está vazia. Vídeo não processado.')
    sys.exit()
elif time2 == '':
    print(f'    ERRO: A coluna time_2 para o vídeo {video_name} está vazia. Vídeo não processado.')
    sys.exit()
else:
    time1 = float(time1.replace(',','.'))
    time2 = float(time2.replace(',','.'))

# converte para frames
frame1 = int(30*time1)
frame2 = int(30*time2)

# lê o vídeo e corta nos pontos de anotação
video_cap = cv2.VideoCapture(args.video)
new_video_path = os.path.join(args.folder, video_prefix+'.mp4')
new_video = cv2.VideoWriter(new_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920,1080))
success = True
frame_count = 0
while success:
    success, frame = video_cap.read()
    frame_count = frame_count+1
    if success and (frame_count>=frame1) and (frame_count<=frame2):
        new_video.write(frame)
    elif not success or (frame_count>frame2):
        break
new_video.release()
