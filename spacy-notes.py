import os
import pandas as pd 
import spacy

""" 
Notes from the video lecutures here ->
https://www.youtube.com/watch?v=Y90BJzUcqlI&index=1&list=PLJ39kWiJXSiz1LK8d_fyxb7FTn4mBYOsD
"""



#------------------------------------------------------------------------------------------------
#---------------------------- SPACY -------------------------------------------------------------
#------------------------------------------------------------------------------------------------

#---------------------------- INIT --------------------------------------------------------------
"""
nlp = spacy.load('en')  # loads the english library
#                         creates nlp object with pipeline from binary library stored on disk
docx = nlp("Some text document eh") # potentially each text entry?

myfile = open("exfile.txt").read()
dic_file = nlp(myfile)
"""


#---------------------------- TOKENIZATION ------------------------------------------------------
"""
#splitting on '.'
for sentence in enumerate(doc_file.sents):
    print(f'{num}: {sentence}')

# each word
[token.text for token in doc]
for token in doc:
    print(token.text)

doc = nlp(u"welcome to the club meow")
for word in doc:
    print(word.text, word.shape, word.shape_) # .shape_ does the XXXXX stuff

doc2 = nlp("Hello hello HELLO HeLLO")
for word in doc2:
    print("Token:", word.text, "| Shape:", word.shape_, "| Alphanumeric:", word.is_alpha)
    #Token: Hello | Shape: Xxxxx
    #Token: hello | Shape: xxxx
    #Token: HELLO | Shape: XXXX
    #Token: HeLLO | Shape: XxXXX
"""



#---------------------------- PARTS of SPEECH TAGGING --------------------------------------------
"""
ex1 = nlp("He drinks a drink")
for word in ex1:
    print(word.text, word.pos_, word.pos) # .pos returns numbers like so
    #He PRON 93
    #drinks VERB 98
    #a DET 88
    #drink NOUN 90

ex2 = nlp("I fish a fish")
for word in ex2:
    print(word.text, word.pos_, word.tag_, word.tag)

    #   I       PRON    PRP     479
    #   fish    VERB    VBP     492
    #   a       DET     DT      460
    #   fish    NOUN    NN      474

print(spacy.explain("DT")) # says what a thing is
print("----------------------")

excercize1 = nlp(u"All the faith he had had had had no effect on the outcome of his life")
for word in excercize1:
    print((word.text, word.tag_, word.pos_))

print("----------------------")
"""


#---------------------------- SYNTACTIC DEPENDANCY -----------------------------------------------
"""
ex3 = nlp("Sally likes sammy")
for word in ex3:
    print((word.text, word.tag_, word.pos_, word.dep_, word.dep))
    #('Sally', 'NNP', 'PROPN', 'nsubj', 425)
    #('likes', 'VBZ', 'VERB', 'ROOT', 512817)
    #('sammy', 'NN', 'NOUN', 'dobj', 412)

print("----------------------")
ex3 = nlp("Sally likes Sammy")
for word in ex3:
    print((word.text, word.tag_, word.pos_, word.dep_, word.dep))
    #('Sally', 'NNP', 'PROPN', 'nsubj', 425)
    #('likes', 'VBZ', 'VERB', 'ROOT', 512817)
    #('Sammy', 'NNP', 'PROPN', 'dobj', 412)
"""


#---------------------------- VISUALIZING ---------------------------------------------------------
"""
from spacy import displacy
displacy.render(ex3, style="dep", jupyter=False) 
"""

#---------------------------- LEMMATIZING ---------------------------------------------------------
"""
# Reduce word to its base/root form
docx = nlp("study studying studious studio student")
for word in docx:
    print(word.text, word.lemma_, word.pos_)
    
    #   study       study       NOUN
    #   studying    study       VERB
    #   studious    studious    ADJ
    #   studio      studio      NOUN
    #   student     student     NOUN

print("--------------------------")
docx2 = nlp("walking walks walk walker Walk WALK")
for word in docx2:
    print(word.text, word.lemma_, word.pos_)
    #walking    walk    VERB
    #walks      walk    VERB
    #walk       walk    NOUN
    #walker     walker  NOUN
    #Walk       walk    NOUN
    #WALK       walk    PROPN

print("---------------------------")
docx3 = nlp("good goods run running runner runny was be were")
for word in docx3:
    print(word.text, word.lemma_, word.pos_)
    #good       good    ADJ
    #goods      good    NOUN
    #run        run     VERB
    #running    run     VERB
    #runner     runner  NOUN
    #runny      runny   NOUN
    #was        be      VERB
    #be         be      VERB
    #were       be      VERB
"""
#--------------------------------------------------------------------
#---------------------------- SPACY ---------------------------------
#--------------------------------------------------------------------