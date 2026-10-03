import os

# Configure todo o novo dataset em ./config.env 

#Primeira Execução:
os.system('python3 new_dataset_video_preprocess.py')
#Objetivo: Processar os vídeos do dataset, incluindo cortes e redimensionamento. O script também pode gerar um arquivo errors.csv com informações sobre vídeos corrompidos ou problemas encontrados.

#Segunda Execução (para garantir a remoção de vídeos corrompidos):
os.system('python3 new_dataset_video_preprocess.py') 
#Objetivo: Reprocessar vídeos, especialmente se houver problemas identificados na primeira execução. Verifique o arquivo errors.csv para confirmar se os problemas foram tratados.

#Terceira Execução (se necessário):
os.system('python3 new_dataset_video_preprocess.py')
#Objetivo: Garantir que todos os vídeos foram processados corretamente e que quaisquer problemas foram corrigidos.

os.system('python3 new_dataset_preprocess.py')
# Gera informações sobre os vídeos a partir de arquivos CSV. Atualiza um dicionário de sinais. Cria um arquivo de verdade de referência (groundtruth) para avaliação.