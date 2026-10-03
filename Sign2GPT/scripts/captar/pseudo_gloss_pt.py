from tqdm import tqdm
import pandas as pd
import spacy
import pickle
from collections import Counter
import os

language_source = "pt_core_news_lg"

manual_lemmas = {
    "A": ["a"], "B": ["b"], "C": ["c"], "D": ["d"], "E": ["e"], "F": ["f"],
    "G": ["g"], "H": ["h"], "I": ["i"], "J": ["j"], "K": ["k"], "L": ["l"],
    "M": ["m"], "N": ["n"], "O": ["o"], "P": ["p"], "Q": ["q"], "R": ["r"],
    "S": ["s"], "T": ["t"], "U": ["u"], "V": ["v"], "W": ["w"], "X": ["x"],
    "Y": ["y"], "Z": ["z"],
    "Dor lombar": ["dor", "lombar"],
    "Dor na perna": ["dor", "perna"],
    "Urinando pouco": ["urinar", "pouco"],
    "Urinando muito": ["urinar", "muito"],
    "Urina com espuma": ["urina", "espuma"],
    "Sangramento nas fezes": ["sangramento", "fezes"],
    "Bebendo": ["beber"],
    "Não lembro": ["não", "lembrar"],
    "Dor na perna direita e esquerda": ["dor", "perna", "direito", "esquerdo"],
    "Piora ao subir escada?": ["piora", "subir", "escada"],
    "Melhorou com medicamento": ["melhorar", "medicamento"],
    "Foi uma vez": ["ser", "uma", "vez"],
    "Foram vários episódios": ["ser", "episódio"],
    "Tem tosse?": ["ter", "tosse"],
    "Uso medicamento": ["usar", "medicamento"],
    "Não sinto dor na barriga": ["não", "sentir", "dor", "barriga"],
    "Não piora ao subir escada": ["não", "piora", "subir", "escada"],
    "Piora ao subir escada": ["piora", "subir", "escada"],
    "Por que você procurou atendimento nesse momento? Houve piora do quadro?": ["por", "que", "você", "procurar", "atendimento", "momento", "houve", "piora", "quadro"],
    "A dor é como se fosse em choque?": ["dor", "ser", "como", "choque"],
    "Dor de um lado da cabeça": ["dor", "um", "lado", "cabeça"],
    "Existe algum fator de piora?": ["existir", "fator", "piora"],
    "Foi um episódio de dor ou foram mais episódios?": ["ser", "episódio", "dor", "ser", "mais", "episódio"],
    "Qual a frequência da dor?": ["qual", "frequência", "dor"],
    "Quando foi o primeiro episódio?": ["quando", "ser", "primeiro", "episódio"],
    "A dor irradia para a nuca": ["dor", "irradiar", "nuca"],
    "A dor irradia para o pescoço": ["dor", "irradiar", "pescoço"],
    "A dor irradia para a mandíbula": ["dor", "irradiar", "mandíbula"],
    "Irradia para o ombro": ["irradiar", "ombro"],
    "Irradia para o braço": ["irradiar", "braço"],
    "Irradia para a barriga": ["irradiar", "barriga"],
    "A dor irradia para as costas": ["dor", "irradiar", "costa"],
    "A dor irradia para a virilha": ["dor", "irradiar", "virilha"],
    "A dor irradia para a coxa": ["dor", "irradiar", "coxa"],
    "A dor irradia para a perna": ["dor", "irradiar", "perna"],
    "Trouxe receita médica": ["trazer", "receita", "médico"],
    "Está com diarreia?": ["estar", "diarreia"],
    "Sente dor na barriga": ["sentir", "dor", "barriga"],
    "Quais bebidas?": ["qual", "bebida"],
    "Quantas bebidas?": ["quanto", "bebida"],
    "Você fez alguma cirurgia? Se sim, qual?": ["você", "fazer", "cirurgia", "sim", "qual"],
    "Qual cor é o catarro?": ["qual", "cor", "catarro"],
    "Qual medicamento?": ["qual", "medicamento"],
    "Quando começou?": ["quando", "começar"]
}

dict_pos = {
    "ADJ": "adjective",
    "ADP": "adposition",
    "ADV": "adverb",
    "AUX": "auxiliary verb",
    "CONJ": "conjunction",
    "CCONJ": "coordinating conjunction",
    "DET": "determiner",
    "INTJ": "interjection",
    "NOUN": "noun",
    "NUM": "numeral",
    "PART": "particle",
    "PRON": "pronoun",
    "PROPN": "proper noun",
    "PUNCT": "punctuation",
    "SCONJ": "subordinating conjunction",
    "SYM": "symbol",
    "VERB": "verb",
    "X": "other",
}

selected_vocab = ["NOUN", "NUM", "ADV", "PRON", "PROPN", "ADJ", "VERB"]


def get_parts_of_speech(sentence):
    if sentence in manual_lemmas:
        lems = manual_lemmas[sentence]
    else:
        doc = nlp(sentence)
        normalized_sentence = [token.lemma_ for token in doc]
        pos_tags = [(token.text, token.pos_) for token in doc]
        lems = []
        for n, p in zip(normalized_sentence, pos_tags):
            part = p[1]
            if part in selected_vocab:
                lems.append(n.lower())
    return lems


nlp = spacy.load(language_source)

csv = "scripts/captar/captar_sentences.csv"

df = pd.read_csv(csv, sep=";")

sentences = df.sentence.values

dict_sentence = {}
all_lens = []
for sentence in tqdm(sentences):
    sentence = sentence.strip()
    lems = get_parts_of_speech(sentence)
    all_lens.extend(lems)
    dict_sentence[sentence] = lems

# fasttext.util.download_model("zh", if_exists="ignore")
# ft = fasttext.load_model("cc.zh.300.bin")

dict_lem_to_id = {lem: i for i, lem in enumerate(list(set(all_lens)))}

# flat_list = [item for sublist in list(dict_sentence.values()) for item in sublist]
# count = Counter(flat_list)

# vector = torch.zeros((len(dict_lem_to_id),300))
# for key, value in tqdm(dict_lem_to_id.items()):
#     vector[value] = torch.tensor(ft.get_word_vector(key))

# dict_id_to_lem = {v:k for k, v in dict_lem_to_id.items()}

print(f"{Counter(all_lens) = }")
print(f"{dict(Counter(all_lens)) = }")
print(f"{dict(dict_sentence) = }")
print(f"{dict_lem_to_id = }")
dict_processed_words = {
    "dict_lem_counter": dict(Counter(all_lens)),
    "dict_sentence": dict(dict_sentence),
    "dict_lem_to_id": dict_lem_to_id,
}

os.makedirs("data/captar", exist_ok=True)
# Save the processed words to a pickle file
with open("data/captar/processed_words.captar_pkl", "wb") as f:
    pickle.dump(dict_processed_words, f)
