import pandas as pd

# carrega os arquivos
groundtruth_df = pd.read_csv("groundtruth.csv")
consistencia_df = pd.read_csv("consistencia_file.csv")

# cria um dicionário para busca rápida
annotation_dict = dict(zip(groundtruth_df["id"], groundtruth_df["annotation"]))

# lista para armazenar VideoIDs com erro
errors = []

for _, row in consistencia_df.iterrows():
    sentence_id = row["Sentence"]
    predicted_gloss = row["Gloss"]

    expected_gloss = annotation_dict.get(sentence_id)

    # compara gloss esperada com a que está na consistência
    if expected_gloss is not None and expected_gloss != predicted_gloss:
        errors.append([
            row["VideoID"], 
            sentence_id, 
            predicted_gloss, 
            expected_gloss
        ])

if errors:
    errors_df = pd.DataFrame(errors, columns=["VideoID", "Sentence", "Gloss", "Expected"])
    errors_df.to_csv("errors.csv", index=False)
    print(f"Arquivo 'errors.csv' criado com {len(errors)} inconsistências.")
else:
    print("Nenhuma inconsistência encontrada. Nenhum arquivo foi criado.")