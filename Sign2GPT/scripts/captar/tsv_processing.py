import glob
import pandas as pd

dataset_anotations = "./datasets_srv/captar-libras_10-04/annotations"

output_path = "data/captar-libras_10-04-annotated/sentence_label.tsv"
sentence_df = pd.read_csv("./scripts/captar/captar_sentences.csv", sep=";")
df = pd.DataFrame()
sentence_dict = {}

for _, row in sentence_df.iterrows():
    sentence_dict[row["id"]] = row["sentence"]

for file in glob.glob(f"{dataset_anotations}/*.csv"):
    print(f"Reading {file}")
    df_temp = pd.read_csv(file, sep="|")
    df_temp["split"] = file.split("/")[-1].split(".")[0]
    df_temp.rename(columns={"id": "name"}, inplace=True)
    df_temp = df_temp[~df_temp["name"].str.contains("_signal")]
    df = pd.concat([df, df_temp], ignore_index=True)

for index, row in df.iterrows():
    name = row["name"]
    if "_signal" in name:
        continue
    sid = name.split("_s")[1].split("_")[0]

    try:
        df.loc[index, "sentence"] = sentence_dict[sid].strip()
        df.loc[index, "sentence_id"] = sid
        df.loc[index, "name"] = name
    except KeyError:
        print(f"KeyError: {sid} not found in sentence_dict {name}")
        continue

df = df[["name", "sentence_id", "sentence", "split"]]

df.to_csv(output_path, sep="\t", index=False, header=True)
print(f"Data saved to {output_path}")
