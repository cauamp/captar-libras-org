import pandas as pd
import os

# caminhos dos arquivos
base_folder = "./split/2025-04-10"
output_folder = "../datasets_/2025-04-10"

train3_df = pd.read_csv("cover/test2.csv")
test2_df = pd.read_csv("cover/test2.csv")

train3_ids = set(train3_df['ID'].values)
test2_ids = set(test2_df['ID'].values)

# processa cada conjunto (train, test, dev)
def process_files(base_folder):
    signals_df = pd.read_csv(os.path.join(base_folder, "signals_file.csv"))
    
    # itera pelos arquivos de entrada (train.csv, test.csv, dev.csv)
    for split in ['train', 'test', 'validate']:
        input_file = os.path.join(base_folder, f"{split}.csv")
        
        # verifica se o arquivo existe
        if not os.path.exists(input_file):
            print(f"Arquivo {input_file} não encontrado. Pulando...")
            continue
        
        # carrega o CSV atual
        split_df = pd.read_csv(input_file)
        
        # lista para armazenar as novas linhas
        extended_rows = []
        
        # processa cada linha
        for _, row in split_df.iterrows():
            sentence_id = row['sentence_id']
            
            # adiciona a linha original
            extended_rows.append(row.to_dict())
            
            # condicional para verificar se deve processar a linha
            if split == 'train' and sentence_id in train3_ids:
                process_row(row, extended_rows, signals_df)
            elif (split == 'test' or split == 'validate') and sentence_id in test2_ids:
                process_row(row, extended_rows, signals_df)
        
        # cria um DataFrame com as linhas estendidas
        extended_df = pd.DataFrame(extended_rows)
        
        # define o caminho para salvar o arquivo processado
        output_file = os.path.join(output_folder, f"{split}.csv")
        
        # salva o arquivo processado
        extended_df.to_csv(output_file, index=False, header=True)
        print(f"Arquivo processado e salvo em: {output_file}")
        
def process_row(row, extended_rows, signals_df):
    video_id = row['video_id']
        
    # filtra as correspondências no signals.csv
    matching_signals = signals_df[signals_df['VideoIDOriginal'].apply(lambda x: video_id.startswith(x))]
    
    for _, signal_row in matching_signals.iterrows():
        # cria uma nova linha estendida
        extended_row = {
            'date': row['date'],  # da linha original
            'sentence_id': row['sentence_id'],  # da linha original
            'person_id': row['person_id'],  # da linha original
            'video_id': signal_row['VideoID'],  # da signals.csv
            'video_path': row['video_path'],  # da linha original
        }
        extended_rows.append(extended_row)

if __name__ == '__main__':
    # processa os arquivos
    process_files(base_folder)