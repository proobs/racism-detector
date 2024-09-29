# https://ner.pythonhumanities.com/03_06_loading_custom_word_vectors.html
import sys
import subprocess
import spacy
import multiprocessing
import csv 
import json
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

model_name = "racism_detector"
word_vectors = "data/word_vectors.txt"
racism_data = "twitter_racism_parsed_dataset.csv"
racism_json = "data/racism.json"

def transform_racism_data(racism_data):
    data = []

    with open(racism_data, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[3] == 'racism':
                data.append(row[2])  # Extracts sentence
        
    return data 

def save_data_to_file():
    data = transform_racism_data(racism_data)

    with open(racism_json, 'w', newline='') as file:
        json.dump(data, file)

def train_model(name):
    with open(racism_json, 'r') as f:
        data = json.load(f)
    
    cores = multiprocessing.cpu_count()
    tokenized_data = [sentence.split() for sentence in data]
    
    # Initialize Word2Vec model
    model = Word2Vec(
        sentences=tokenized_data,
        min_count=5,
        window=2,
        size=500,
        sample=6e-5,
        alpha=0.03,
        min_alpha=0.0007,
        negative=20,
        workers=cores - 1
    )
    
    # Build vocabulary and train the model
    model.train(data, total_examples=len(data), epochs=30)
    
    # Save the model
    model.save(f'data/word_vec/{name}.model')
    model.wv.save_word2vec_format(f'data/word_vec/{name}.txt')


def similarity(text):
    model = KeyedVectors.load_word2vec_format(f"data/word_vec/{model_name}.txt", binary=False)
    return model.most_similar(positive=[text])


save_data_to_file()
train_model(model_name)
print(similarity("monkey bot"))
print(similarity("i like air"))

    
