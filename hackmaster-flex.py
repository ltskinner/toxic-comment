"""
Script from:
Learning TensorFlow: A guide to building deep learning systems
Tom Hope, Yehezkel S. Resheff & Ital Lieder
"""

import os
import numpy as np 
import pandas as pd
import tensorflow as tf 


"""
# Preprocessing Zero Padding..... lame as shit but I suppose its what you have todo to get consistancy

digit_to_word_map = {1:"One", 2:"Two", 3:"Three", 4:"Four", 5:"Five", 6:"Six", 7:"Seven", 8:"Eight", 9:"Nine"}

digit_to_word_map[0] = "PAD"

even_sentences = []
odd_sentences = []
seqlens = []

for i in range(10000):
    rand_seq_len = np.random.choice(range(3,7))
    seqlens.append(rand_seq_len)
    rand_odd_ints = np.random.choice(range(1, 10, 2), rand_seq_len)
    rand_even_ints = np.random.choice(range(2, 10, 2), rand_seq_len)

    #Padding
    if rand_seq_len<6:
        rand_odd_ints = np.append(rand_odd_ints, [0]*(6-rand_seq_len))
        rand_even_ints = np.append(rand_even_ints, [0]*(6-rand_seq_len))
    
    even_sentences.append(" ".join([digit_to_word_map[r] for r in rand_even_ints]))
    odd_sentences.append(" ".join([digit_to_word_map[r] for r in rand_odd_ints])) #flipped even and odd for some reason???

data = even_sentences + odd_sentences
seqlens*=2


print(even_sentences[0:6])
print("----------------")
print(odd_sentences[0:6])
print("----------------")
print(seqlens[0:6]) # hold onto seqlens to feed to model to tell when to stop processing over noise?


# Mapo from words to indicies
word2index_map = {}
index = 0
for sent in data:
    for word in sent.lower().split():
        if word not in word2index_map:
            word2index_map[word] = index
            index += 1

# Inverse map
index2word_map = {index: word for word, index in word2index_map.items()}
vocabulary_size = len(index2word_map)
"""
"""
labels = [1]*10000 + [0]*10000
for i in range(len(labels)):
    label = labels[i]
    one_hot_encoding = [0]*2
    one_hot_encoding[label] = 1
    labels[i] = one_hot_encoding

data_indecies = list(range(len(data)))
np.random.shuffle(data_indecies)
data = np.array(data)[data_indecies]

labels = np.array(labels)[data_indecies]
seqlens = np.array(seqlens)[data_indecies]
train_x = data[:10000]
train_y = labels[:10000]
train_seqlens = seqlens[:10000]

test_x = data[10000:]
test_y = labels[10000:]
test_seqlens = seqlens[10000:]

def get_sentence_batch(batch_size, data_x, data_y, data_seqlens):
    instance_indecies = list(range(len(data_x)))
    np.random.shuffle(instance_indecies) # not popping, or concerned about seeing same data more than once, 
    #print("Indecies:", instance_indecies)
    batch = instance_indecies[:batch_size] 
    x = [[word2index_map[word] for word in data_x[i].lower().split()] for i in batch]
    y = [data_y[i] for i in batch]
    seqlens = [data_seqlens[i] for i in batch]
    print("x:", len(x), "y:", len(y))
    return x, y, seqlens
"""

#----------------------------------------------------------------------------------------------------------------------------------

import spacy
nlp = spacy.load('en')

#print("[+] Imports Complete [+]")

what = "train"
df = pd.read_csv(os.getcwd() + "\\" + what + ".csv")
#df = pd.read_csv(os.getcwd() + "\\" + what + "-sample.csv", nrows=5000)
#print(len(df["comment_text"]))

#batch_size = 128
#embedding_dimension = 64
#num_classes = 2
hidden_layer_size = 32
#times_steps = 6
#element_size = 1

#num_layers = 3
num_classes = 6 #6 #2
batch_size = 32
post_size = seq_len = 128 # times_steps
vec_size = 300 # embedding_dimension
#n_hidden = 512 # number of units in RNN cell
#learning_rate = .001



train_x = list(df["comment_text"])
#train_y = list(zip(df["toxic"], df["insult"]))
train_y = list(zip(df["toxic"], df["severe_toxic"], df["obscene"], df["threat"], df["insult"], df["identity_hate"]))
print("[+] Imports Complete [+]")


tdf = pd.read_csv(os.getcwd() + "\\test.csv", nrows=100)
names = list(tdf["id"])
test_x = list(tdf["comment_text"])

tdft = pd.read_csv(os.getcwd() + "\\test_labels.csv", nrows=100)
test_y = list(zip(tdft["toxic"], tdft["severe_toxic"], tdft["obscene"], tdft["threat"], tdft["insult"], tdft["identity_hate"]))

def vector_inspector(word):

    if np.isnan(np.sum(word.vector)):
        print("HUGE PROBLEM:", word.text)


def meaterizer(train_x, nlp):
    frame = []
    seqlens = []
    for i in train_x:
        buff = []
        doc = nlp(i)
        """
        for word in doc:
            vector_inspector(word)
        """

        if len(doc) == post_size: 
            #print("Even")
            seqlens.append(post_size)

            
            for word in doc:
                buff.append(word.vector)
            

        elif len(doc) > post_size:
            #print("Long")
            seqlens.append(post_size)
            bk = post_size/2
            condenser = []
            cond = np.zeros(vec_size)
            for word in doc[0:bk]:
                buff.append(word.vector)
            for word in doc[bk+1:-bk]: # could optimize no doubt
                condenser.append(word)
            for word in condenser:
                cond = np.add(cond, word.vector)
            buff.append(cond)
            for word in doc[-(bk-1):]:
                buff.append(word.vector)


        elif len(doc) < post_size:
            #print("Short")
            seqlens.append(len(doc))

            for word in doc:
                buff.append(word.vector)

            while len(buff) < post_size:
                buff.append(np.zeros(vec_size))
            

        #print(len(buff))
            

        frame.append(buff)
    """
    for post in frame:
        for word in post:
            if len(word) != 300:
                print("EVEN MORE MASSIVE PROBLEM")
    """

    return frame, seqlens


#print("Max:", max(sizes))
#print("AVG:", int(sum(sizes)/len(sizes)))

def get_batch(batch_size, data_x, data_y, nlp):
    #print("Fre$h batch")
    instance_indecies = list(range(len(data_x)))
    np.random.shuffle(instance_indecies)
    batch = instance_indecies[:batch_size] 

    x, seqlens = meaterizer([data_x[i] for i in batch], nlp)
    y = [data_y[i] for i in batch]
    #print("Successfully made it.")
    return np.nan_to_num(np.array(x)), np.nan_to_num(np.array(y)), np.nan_to_num(np.array(seqlens))




def get_test(data, nlp):
    x, seqlens = meaterizer([data], nlp)
    return np.array(x), np.array(seqlens)




#----------------------------------------------------------------------------------------------------------------------------------










#_inputs = tf.placeholder(tf.int32, shape=[batch_size, post_size])
_labels = tf.placeholder(tf.float32, shape=[None, num_classes]) # batch_size
_seqlens = tf.placeholder(tf.int32, shape=[None]) # batch_size

embed = tf.placeholder(tf.float32, shape=[None, seq_len, vec_size])

# --------------------------------------------- LSTM Stuff ---------------------------------------------------

with tf.variable_scope("lstm"):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size, forget_bias=1.0) # Basic LSTM Cell yo
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, embed, sequence_length=_seqlens, dtype=tf.float32)

weights = { 'linear_layer': tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes], mean=0, stddev=0.01))}

biases = { 'linear_layer':tf.Variable(tf.truncated_normal([num_classes], mean=0, stddev=0.01)) }
# ---------------------------------------------------------------------------------------------------------

# Extract the last relevant output and use in linear layer
#final_output = tf.matmul(states[1], weights["linear_layer"]) + biases["linear_layer"]
final_output = tf.add(tf.matmul(states[1], weights["linear_layer"]), biases["linear_layer"])
pred = tf.nn.sigmoid_cross_entropy_with_logits(logits=final_output, labels=_labels)
#pred = tf.nn.sigmoid(labels=_labels, logits=final_output)
#softmax = tf.nn.softmax_cross_entropy_with_logits(logits=final_output, labels=_labels)
#y = tf.nn.softmax(final_output)
cross_entropy = tf.reduce_mean(pred) # cost, loss

# ---------------------------- Training Embeddings and the LSTM Classifier ------------------------------------
"""
# Original
train_step = tf.train.RMSPropOptimizer(0.001, 0.9, centered=True).minimize(cross_entropy)
"""
#train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)
#grads_and_vars = train_Step.compute_gradients(cross_entropy, )

optimizer = tf.train.GradientDescentOptimizer(.01)
gradients, variables = zip(*optimizer.compute_gradients(cross_entropy))
gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
train_step = optimizer.apply_gradients(zip(gradients, variables)) #https://stackoverflow.com/questions/36498127/how-to-apply-gradient-clipping-in-tensorflow/43486487

#train_step = tf.abs([1])
#train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(_labels, 1), tf.argmax(final_output, 1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))*100
"""
correct = tf.equal(final_output, tf.equal(_labels,1.0))
accuracy = tf.reduce_mean( tf.cast(correct, 'float') )
"""
#---------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ct = 0
    for step in range(250): #arb num of training epochs i reckon
        """
        print("count:", ct)
        ct += 1
        w = sess.run(weights)
        print(w)
        b = sess.run(biases)
        print(b)
        """
        x_batch, y_batch, seqlen_batch = get_batch(256, train_x, train_y, nlp)
        """
        for i in x_batch:
            for j in i:
                for k in np.isnan(j):
                    if k == True:
                        print("HUGE PROBLEM X")
        for i in y_batch:
            for j in np.isnan(i):
                if j == True:
                    print("HUGE PROBLEM Y")
        """

        #print("X:", x_batch[0])
        #print("Y:", y_batch[0])
        #print("new xbatch ------>")
        #print(x_batch)
        """
        for i in x_batch:
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print(i)
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        """
        #print("Pre sess.run, shapes:")
        #print(x_batch.shape)
        #print(y_batch.shape)
        #print(seqlen_batch.shape)
        
        sess.run(train_step, feed_dict={embed:x_batch, _labels:y_batch, _seqlens:seqlen_batch})
        sess.run(train_step, feed_dict={embed:x_batch, _labels:y_batch, _seqlens:seqlen_batch})
        sess.run(train_step, feed_dict={embed:x_batch, _labels:y_batch, _seqlens:seqlen_batch})
        """
        print(fin)
        print("-----------")
        print(state)
        print("-----------")
        """
        #print("Pre step")
        if step % 5 == 0:
            acc, aut = sess.run([accuracy, final_output], feed_dict={embed:x_batch, _labels:y_batch, _seqlens:seqlen_batch})
            
            for i in range(5):
                print(y_batch[i], aut[i])
            
            print("Accuracy at %d: %.5f" % (step, acc))
        

    #----------------------------------
    
    for i in range(len(test_x)):
        #print(i)
        in_x, sq_len = get_test(test_x[i], nlp)
        #print(in_x)
        #print(sq_len)
        
        #print(names[i])
        #print("x_len", in_x)
        #print("x_len", len(in_x[0]))
        #print("sq:", sq_len)
        
        
        #classification = sess.run(y, feed_dict={embed: in_x, _seqlens:sq_len})
        #print(classification.shape)
        #print(classification)
        
        output_example = sess.run(final_output, feed_dict={embed:in_x, _seqlens:sq_len})
        print(test_y[i], "|", output_example)
