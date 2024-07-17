import sys
sys.path.append('./')
import nltk
nltk.download('punkt')
import os
import pandas as pd
import json
from keyphrase_vectorizers import KeyphraseCountVectorizer
from keybert import KeyBERT
from tqdm import tqdm


def extract_keywords(model, vectorizer, dataset_name, dataset):
    for i, dialogue in tqdm(enumerate(dataset['dialogue']), total=len(dataset['dialogue'])):
        results = model.extract_keywords(docs=dialogue, vectorizer=vectorizer)
        kws = [pair[0] for pair in results]
        dataset['kws'][i] = kws

    with open(f'data/{dataset_name}_with_kws.json', 'w') as json_file:
        json.dump(dataset, json_file, indent=4)


kw_model =  KeyBERT()
vectorizer = KeyphraseCountVectorizer()

for file_name in os.listdir('data/'):
    if file_name.endswith('raw.json'):
        with open('data/'+file_name, 'r') as json_file:
            data = json.load(json_file)
            if 'samsum' in file_name:
                df = pd.DataFrame(data)
                data = {
                    'id': df['id'].tolist(),
                    'dialogue': df['dialogue'].tolist(),
                    'summary': df['summary'].tolist(),
                    'kws': [None for _ in range(len(df['id'].tolist()))]
                }
            else:
                if 'test' in file_name:
                    df = pd.DataFrame(data)
                    data = {
                        'id': df['id'].tolist(),
                        'dialogue': df['dialogue'].tolist(),
                        'summary': df['summary'].tolist(),
                        'summary2': df['summary2'].tolist(),
                        'summary3': df['summary3'].tolist(),
                        'kws': [None for _ in range(len(df['id'].tolist()))]
                    }
                else:
                    df = pd.DataFrame(data)
                    data = {
                        'id': df['id'].tolist(),
                        'dialogue': df['dialogue'].tolist(),
                        'summary': df['summary'].tolist(),
                        'kws': [None for _ in range(len(df['id'].tolist()))]
                    }

            extract_keywords(kw_model, vectorizer, file_name.split('_raw')[0], data)
        