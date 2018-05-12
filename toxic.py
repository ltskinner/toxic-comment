
import os
import pandas as pd 
import spacy

what = "train.csv"
df = pd.read_csv(os.getcwd() + "\\" + what, nrows=50)
# ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
raw = "string encoding to unicode or something"


#--------------------------------------------------------------------
#---------------------------- SPACY ---------------------------------
#--------------------------------------------------------------------

nlp = spacy.load('en')  # loads the english library
#                         creates nlp object with pipelinen from binary library stored on disk
docx = nlp("Some text document eh") # potentially each text entry?
#--------------------------------------------------------------------
#---------------------------- SPACY ---------------------------------
#--------------------------------------------------------------------






#       str() goes to unicode i reckon
doc = nlp(str(raw)) # First splits on whitespace to make each token
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)

print("----------------------------------------------")
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

print("----------------------------------------------")
for token in doc:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)