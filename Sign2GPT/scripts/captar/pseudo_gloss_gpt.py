import os
import pickle
import pandas as pd
from tqdm import tqdm
from collections import Counter

df_lemmas = pd.read_csv("data/captar-libras_10-04-gpt/lemmas.csv", sep=";")
df_sentences = pd.read_csv("scripts/captar/captar_sentences.csv", sep=";")

id_to_sentence = {
    id_: sentence.strip()
    for id_, sentence in zip(df_sentences['id'], df_sentences['sentence'])
}

dict_sentence = {}
all_lens = []

for _, row in tqdm(df_lemmas.iterrows(), total=len(df_lemmas)):
    id_ = row['id']
    lemma_sentence = row['sentence'].strip()
    lems = [token for token in lemma_sentence.split(" ") if token]
    all_lens.extend(lems)
    
    final_sentence = id_to_sentence.get(id_, lemma_sentence)
    dict_sentence[final_sentence] = lems

dict_lem_to_id = {lem: i for i, lem in enumerate(list(set(all_lens)))}

print(f"{Counter(all_lens) = }")
print(f"{dict(Counter(all_lens)) = }")
print(f"{dict(dict_sentence) = }")
print(f"{dict_lem_to_id = }")

dict_processed_words = {
    "dict_lem_counter": dict(Counter(all_lens)),
    "dict_sentence": dict(dict_sentence),
    "dict_lem_to_id": dict_lem_to_id,
}

#os.makedirs("data/captar", exist_ok=True)
with open("data/captar-libras_10-04-gpt/processed_words.captar_pkl", "wb") as f:
    pickle.dump(dict_processed_words, f)
