import pandas as pd
import glob 
df = pd.DataFrame()
file_paths = glob.glob('../datasets_srv/captar-libras_16-10/annotations/*.csv') 

sinais_treino = set()

for file_path in file_paths: 
    # Concatenar o arquivo CSV com delimitador ajustado
    df = pd.concat([df, pd.read_csv(file_path, delimiter='|')])


    # Garante que a coluna 'annotation' existe no dataset
    if 'annotation' not in df.columns:
        raise ValueError("A coluna 'annotation' não foi encontrada no arquivo.")

    # Realiza o split nos valores da coluna 'annotation' para separar os sinais
    sinais = df['annotation'].str.split()

    # Conta os sinais únicos
    sinais_unicos = set(sinal for sublist in sinais.dropna() for sinal in sublist)

    if 'train' in file_path:
        sinais_treino = sinais_treino.union(sinais_unicos)
    else:
        for sinal in sinais_unicos:
            if sinal not in sinais_treino:
                print(f'Sinal fora do treino: {sinal}')
    # Número total de sinais únicos
    print(file_path)
    print(f'{len(sinais_unicos) = } \n')
