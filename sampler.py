

import os
import numpy as np
import pandas as pd
import tensorflow as tf 

from datetime import datetime

import spacy
nlp = spacy.load('en')


doc1 = nlp("test one for what the heck a token is and if I can use a pretrained monster")
for token in doc1:
    print(token.text, ":", token.orth)


print("[+] Imports Complete [+]")

what = "train"
df = pd.read_csv(os.getcwd() + "\\" + what + "-sample.csv", nrows=5000)
#print(len(df["comment_text"]))

train_x = list(df["comment_text"])
train_y = list(zip(df["toxic"], df["insult"]))


vocab = {}

start = datetime.now()
for i in train_x:
    ea = nlp(i)
    for tok in ea:
        if tok.orth not in vocab.keys():
            vocab[tok.text] = tok.vector


print("Size for 2500:", len(vocab.keys()))
print(datetime.now() - start)

# 12 seconds, will take ~ 12 min to convert entire set to dictionary, not sure if worth....
# interesting notes on storing vectors on gpu though
# https://spacy.io/usage/vectors-similarity

