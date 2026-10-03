import pandas as pd
import glob
import json
import re
from tqdm import tqdm
import csv
from spelled_sentences import sentencas_soletradas
from manual_correction import sentencas_alteradas
from pathlib import Path
import shutil
from datetime import datetime as dt

# Caminho para o arquivo Excel
file_path = '[Captar-Libras] Glosas 2024.xlsx'

# Lista de arquivos JSON na pasta ELAN-out
elan_json = glob.glob('ELAN-out/*.json') 

# Lê todas as planilhas do Excel, ignorando a primeira linha
all_sheets = pd.read_excel(file_path, sheet_name=None, skiprows=0)

# Caminho para anotaçoes orignais
source_dir = Path('./ELAN-annotation/') 

# Caminho para backup das anotações inconsistentes encontradas 
destination_dir = Path('/srv/projects2/captarlibras_finep/annotation-inconsistency')
backup = False #Ativa o backup

# Remove a planilha 'Página1' se ela existir
if 'Página1' in all_sheets:
    del all_sheets['Página1']
    
result_dict = {}
# Processa cada planilha
for sheet_name, df in all_sheets.items():
    for i in range(len(df)):
        key = str(df.iloc[i, 0]).strip()  # Converte para string se necessário
        value = df.iloc[i, 3]  
        if key in sentencas_soletradas:
            result_dict[key] = sentencas_soletradas[key]
        else:
            if key == '148':
                for i in range(1, 27):
                    result_dict[key + f'.{i}'] = str(chr(i + 64))
            elif key == '170':  
                for i in range(11):
                    result_dict[key + f'.{i}'] = str(i)
            elif key == '  P02          Quando começou?   ':  
                result_dict['P2'] = str(value)
            else:
                if key[0] == 'P':
                    key = key[0] + str(float(key[1:])).replace('.0', '')
                    if key in sentencas_soletradas:
                        result_dict[key] = sentencas_soletradas[key]
                        continue
                value = ' '.join(value.split()).strip()
                result_dict[key] = str(value)

consistencia = []  
inconsistencia = []
inconsistencia_xlsx = []
signals = []  
signal_video_dropped = 0
# Processa cada arquivo JSON
for json_file_path in tqdm(elan_json, desc='Consultando jsons'):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    video_path = data.get('video_path')
    match = re.search(r'\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2}', video_path)
    if '_p7' in data.get('video_name'):
        continue
    
    if match:
        video_name = video_path[match.start():]
    else: 
        video_name =  data.get('video_name')
    
    video_id =  data.get('video_name')

    idx_match = re.search(r'_s(.*\d)', video_path)

    if idx_match:
        idx = idx_match.group(0).replace('_', "").split('b')[0]   
        if idx[0] == 's':
            idx = idx[1:]
    
    if idx[0].upper() == 'P':
        first = idx[0].upper()
        end = str(float(idx[1:])).replace('.0', '')
        idx = f"{first}{end}"
        
    
    gloss_annotation = str(data.get('gloss_annotation'))
    if idx in sentencas_soletradas:
        gloss_annotation = gloss_annotation.replace('-', " ")
        gloss_annotation = gloss_annotation.replace('OU', "O U")
        if idx == 'P11.4':
            gloss_annotation = gloss_annotation.replace('AO LONGO DO TEMPO', "AO-LONGO-DO-TEMPO")
        if idx == 'P9.7':
            gloss_annotation = gloss_annotation.replace('TOMAR REMÉDIO', "TOMAR-REMÉDIO")
        if idx == 'P14':
            gloss_annotation = gloss_annotation.replace('SECREÇÃO NASAL', "SECREÇÃO-NASAL")
        if idx == 'P15.3':
            gloss_annotation = gloss_annotation.replace('TER NÃO', "TER-NÃO")
            
            
    # Verifica a existência das chaves antes de acessar
    if idx in result_dict:
        if result_dict[idx] != gloss_annotation:
            inconsistencia.append([video_name, idx, gloss_annotation, result_dict[idx]])
        else:
            if idx in sentencas_alteradas:
                gloss_annotation = sentencas_alteradas[idx]
                
            times = data.get('annotations')[0]
            time_1 = times['start_time']
            time_2 = times['end_time']
            
            match = re.search(r"(\d{2})-(\d{2})-(\d{4})_(\d{2})-(\d{2})-(\d{2})", video_name)
            datetime = f"{match.group(3)}-{match.group(2)}-{match.group(1)} {match.group(4)}:{match.group(5)}:{match.group(6)}"

            consistencia.append([video_id, datetime, idx, gloss_annotation, time_1, time_2] )

            spelled_glosses = {'O-V-O', 'A-V-C', 'M-U-C-O', 'S-I', 'F-U-N-G-O', 'G-I-N', 'T-R-I-G-O', 'O-U', 'S-I', 'A-R', 'P-E', 'P-U-S', 'F-U-N-G-O-S'}
            
            corrected_tokens = sentencas_alteradas[idx].split() if idx in sentencas_alteradas else []
            
            # adiciona o início e o fim de cada sinal desse vídeo num csv separado
            for i, annotation in enumerate(data.get('annotations')[1:]):
                gloss = annotation['value']
                
                if gloss in spelled_glosses:
                    signal_video_dropped += 1
                    continue
                
                start_time = annotation['start_time']
                end_time = annotation['end_time']
                
                signal_id = f"{video_id}_signal{i}"
                
                if idx in sentencas_alteradas and i < len(corrected_tokens):
                    gloss = corrected_tokens[i]
                    
                signals.append([signal_id, video_id, datetime, idx, gloss, start_time, end_time])

    else:
        inconsistencia_xlsx.append([video_name, idx, gloss_annotation, 'NOT FOUND IN EXCEL'])

print(f'Videos dropped by gloss spelling: {signal_video_dropped}')
# Escreve as inconsistências em um arquivo de texto
header = ['VideoID', 'Sentence', 'Elan', 'Gloss']
today = dt.today().strftime('%d-%m-%Y')

with open('inconsistencia_file.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # Escreva o cabeçalho no arquivo
    writer.writerow(header)
    
    # Escreva as linhas de 'inconsistencia' no arquivo
    for line in inconsistencia:
        writer.writerow(line)
with open('inconsistencia_xlsx_file.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # Escreva o cabeçalho no arquivo
    writer.writerow(header)
    
    # Escreva as linhas de 'inconsistencia' no arquivo
    for line in inconsistencia_xlsx:
        writer.writerow(line)
        if backup:
            source_file = source_dir / f"{line[0].split('.')[0]}.eaf"
            destination_file = destination_dir / f"{line[0].split('.')[0]}_{today}.eaf"
            if source_file.exists():
                try:
                    shutil.copy(source_file, destination_file)
                    print(f"Arquivo copiado: {destination_file}")
                except Exception as e:
                    print(f"Erro ao copiar {source_file}: {e}")
            else:
                print(
                    f"Arquivo não encontrado: {source_file}")
        
consistencia_header = ['VideoID', 'Datetime', 'Sentence', 'Gloss', 'time_1', 'time_2']
with open('consistencia_file.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # Escreva o cabeçalho no arquivo
    writer.writerow(consistencia_header)
    
    # Escreva as linhas de 'consistencia' no arquivo
    for line in consistencia:
        writer.writerow(line)

signals_header = ['VideoID', 'VideoIDOriginal', 'Datetime', 'Sentence', 'Gloss', 'time_1', 'time_2']
with open('signals_file.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    writer.writerow(signals_header)
    
    for line in signals:
        writer.writerow(line)

all_rows = []
for line in consistencia:
    all_rows.append(line[:4] + line[4:])

for line in signals:
    all_rows.append(line[:1] + line[2:])

with open('checked.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    writer.writerow(consistencia_header)
    
    for row in all_rows:
        writer.writerow(row)