import pandas as pd
import math

# caminhos
base_folder = "../ELAN/"
excel_path = "./split/[Captar-Libras] Lista de sentenças e pessoas informantes.xlsx"
consistencia_path = base_folder + "consistencia_file.csv"

# lê os arquivos
df = pd.read_excel(excel_path, sheet_name="Anotações")
consistencia_df = pd.read_csv(consistencia_path)

def float_conv(series):
    return pd.to_numeric(series.astype(str).str.replace(',', '.'), errors='coerce')

df['time_1'] = float_conv(df['time_1'])
df['time_3'] = float_conv(df['time_3'])

df = df[df['time_1'].notna() & df['time_3'].notna()]

# cria rough_notes.csv com colunas do checked + tempos da planilha
rough_notes = consistencia_df[['VideoID', 'Datetime', 'Sentence', 'Gloss']].copy()

# adiciona time_1 e time_3 a partir da planilha Excel
def get_times(video_id):
    match = df[df['video_id'].str.startswith(video_id)]
    if not match.empty:
        first = match.iloc[0]
        return pd.Series([int(first['time_1']*1000), int(first['time_3']*1000)])
    return pd.Series([None, None])

rough_notes[['time_1', 'time_2']] = rough_notes['VideoID'].apply(get_times)
rough_notes = rough_notes[rough_notes['time_1'].notna() & rough_notes['time_2'].notna()]
rough_notes[['time_1', 'time_2']] = rough_notes[['time_1', 'time_2']].astype('Int64')

# salva rough_notes.csv
rough_notes.to_csv(base_folder + "rough_notes.csv", index=False)

diffs = []

# cria diff.csv com diferenças de tempo em frames
for _, row in consistencia_df.iterrows():
    video_id = row['VideoID']
    time1csv = row['time_1']
    time2csv = row['time_2']
    time1csv = math.floor(time1csv/1000 * 30)
    time2csv = math.ceil(time2csv/1000 * 30)

    match = df[df['video_id'].str.startswith(video_id)]

    if not match.empty:
        first_match = match.iloc[0]
        time1_df = first_match['time_1']
        time3_df = first_match['time_3']

        if pd.notna(time1_df) and pd.notna(time3_df):
            frame1excel = math.floor(time1_df * 30)
            frame3excel = math.ceil(time3_df * 30)
            time1 = time1csv - frame1excel
            time3 = frame3excel - time2csv
                      
            diffs.append({
                'VideoID': video_id,
                'diff_start': time1,
                'diff_end': time3,
            })

diff_df = pd.DataFrame(diffs)
diff_df.to_csv(base_folder + "diff.csv", index=False)

print("Arquivos 'rough_notes.csv' e 'diff.csv' criados com sucesso.")
