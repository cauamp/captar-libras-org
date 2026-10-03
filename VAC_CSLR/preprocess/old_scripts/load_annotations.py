import os
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv

load_dotenv()
GOOGLE_SHEETS_JSON = os.getenv('GOOGLE_SHEETS_JSON')
GOOGLE_SHEETS_KEY = os.getenv('GOOGLE_SHEETS_KEY')

def load_annotations():
    # lê o arquivo de anotações
    scope = ["https://spreadsheets.google.com/feeds",
             'https://www.googleapis.com/auth/spreadsheets',
             "https://www.googleapis.com/auth/drive.file",
             "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_SHEETS_JSON, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(GOOGLE_SHEETS_KEY)
    annotations_sheet = sheet.get_worksheet(2)

    # transforma em dataframe
    annot_data = annotations_sheet.get_all_values()
    annotations = pd.DataFrame(annot_data[1:], columns=annot_data[0])

    # baixa em csv
    annotations_path = "annotations.csv"
    annotations.to_csv(annotations_path, index=False)

    return annotations_path
    
def load_sentences(version='new'):
   # lê o arquivo de sentenças
    scope = ["https://spreadsheets.google.com/feeds",
             'https://www.googleapis.com/auth/spreadsheets',
             "https://www.googleapis.com/auth/drive.file",
             "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_SHEETS_JSON, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(GOOGLE_SHEETS_KEY)
    
    if version == 'new':
        sentences_sheet = sheet.get_worksheet(1)
    else:
        sentences_sheet = sheet.get_worksheet(4)

    # transforma em dataframe
    sentences_data = sentences_sheet.get_all_values()
    sentences = pd.DataFrame(sentences_data[1:], columns=sentences_data[0])

    # baixa em csv
    sentences_path = "sentences.csv"
    sentences.to_csv(sentences_path, index=False)

    return sentences_path

if __name__ == "__main__":
    annotations_data = load_annotations()
    print(annotations_data)
