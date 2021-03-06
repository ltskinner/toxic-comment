
import os
import csv

import numpy as np 
import pandas as pd
import tensorflow as tf 

from datetime import datetime
start = datetime.now()
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
hidden_layer_size = 515
#times_steps = 6
#element_size = 1

#num_layers = 3
num_classes = 6 #6 #2
batch_size = 32
post_size = seq_len = 300 # times_steps
vec_size = 300 # embedding_dimension
#n_hidden = 512 # number of units in RNN cell
#learning_rate = .001



train_x = list(df["comment_text"])
#train_y = list(zip(df["toxic"], df["insult"]))
train_y = list(zip(df["toxic"], df["severe_toxic"], df["obscene"], df["threat"], df["insult"], df["identity_hate"]))
print("[+] Imports Complete [+]")

del df

ones_x = []
ones_y = []

for i in range(len(train_x)):
    if 1 in train_y[i]:
        ones_x.append(train_x[i])
        ones_y.append(train_y[i])

"""
tdf = pd.read_csv(os.getcwd() + "\\test.csv") #, nrows=100)
names = list(tdf["id"])
test_x = list(tdf["comment_text"])

tdft = pd.read_csv(os.getcwd() + "\\test_labels.csv", nrows=100)
test_y = list(zip(tdft["toxic"], tdft["severe_toxic"], tdft["obscene"], tdft["threat"], tdft["insult"], tdft["identity_hate"]))
"""

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

def get_random_batch(batch_size, data_x, data_y, nlp):
    #print("Fre$h batch")
    instance_indecies = list(range(len(data_x)))
    np.random.shuffle(instance_indecies)
    batch = instance_indecies[:batch_size] 

    x, seqlens = meaterizer([data_x[i] for i in batch], nlp)
    y = [data_y[i] for i in batch]
    #print("Successfully made it.")
    return np.nan_to_num(np.array(x)), np.nan_to_num(np.array(y)), np.nan_to_num(np.array(seqlens))

def get_seq_batch(i, batch_size, data_x, data_y, nlp):
    #print("Fre$h batch")
    instance_indecies = list(range(batch_size))

    x, seqlens = meaterizer(data_x[i:i+batch_size], nlp)
    y = data_y[i:i+batch_size]
    #print("Successfully made it.")
    return np.nan_to_num(np.array(x)), np.nan_to_num(np.array(y)), np.nan_to_num(np.array(seqlens))



def get_test(data, nlp):
    x, seqlens = meaterizer(data, nlp)
    return np.nan_to_num(np.array(x)), np.nan_to_num(np.array(seqlens))


def csvWriteRow(yuuge_list, filename):
    if '.csv' not in filename:
        filename = filename + '.csv'

    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in yuuge_list:
            writer.writerow(line)
    
    print('[+] Successfully exported data to', filename, '[+]\n')


#----------------------------------------------------------------------------------------------------------------------------------







h = ['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
outframe = [h]

#_inputs = tf.placeholder(tf.int32, shape=[batch_size, post_size])
_labels = tf.placeholder(tf.float32, shape=[None, num_classes], name='labels') # batch_size
_seqlens = tf.placeholder(tf.int32, shape=[None], name='seqlens') # batch_size

embed = tf.placeholder(tf.float32, shape=[None, seq_len, vec_size], name='embed')

# --------------------------------------------- LSTM Stuff ---------------------------------------------------

with tf.variable_scope("lstm"):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size, forget_bias=1.0) # Basic LSTM Cell yo
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, embed, sequence_length=_seqlens, dtype=tf.float32)

weights =  tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes], mean=0, stddev=0.01), name="weights")

biases = tf.Variable(tf.truncated_normal([num_classes], mean=0, stddev=0.01), name="biases") 
# ---------------------------------------------------------------------------------------------------------

# Extract the last relevant output and use in linear layer
#final_output = tf.matmul(states[1], weights["linear_layer"]) + biases["linear_layer"]
final_output = tf.add(tf.matmul(states[1], weights), biases)
pred = tf.nn.sigmoid_cross_entropy_with_logits(logits=final_output, labels=_labels)
final_output_sig = tf.sigmoid(final_output, name="final_output_sig")
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
"""
optimizer = tf.train.GradientDescentOptimizer(.01)
gradients, variables = zip(*optimizer.compute_gradients(cross_entropy))
gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
train_step = optimizer.apply_gradients(zip(gradients, variables)) #https://stackoverflow.com/questions/36498127/how-to-apply-gradient-clipping-in-tensorflow/43486487
"""
#train_step = tf.abs([1])
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(_labels, 1), tf.argmax(final_output, 1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))*100
"""
correct = tf.equal(final_output, tf.equal(_labels,1.0))
accuracy = tf.reduce_mean( tf.cast(correct, 'float') )
"""

saver = tf.train.Saver()
#---------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------

batch = 256
times = 6000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ct = 0
    ran_ct = 625
    initct = 256
    for step in range(initct): # had [1, 0, 0, 0, 0, 0] figured out around 200
        x_batch, y_batch, seqlen_batch = get_random_batch(batch, ones_x, ones_y, nlp)
        sess.run(train_step, feed_dict={embed:x_batch, _labels:y_batch, _seqlens:seqlen_batch})
        if step % 5 == 0:
            acc, aut = sess.run([accuracy, final_output_sig], feed_dict={embed:x_batch, _labels:y_batch, _seqlens:seqlen_batch})
            
            for i in range(5):
                print(y_batch[i], aut[i])
            
            print(str(step) + '/' + str(initct) + " R-init RAN: Accuracy at %d: %.5f" % (step, acc))

    # Sequenced 
    for step in range(0, len(train_x), batch): #arb num of training epochs i reckon
        #x_batch, y_batch, seqlen_batch = get_random_batch(batch, train_x, train_y, nlp)
        x_batch, y_batch, seqlen_batch = get_seq_batch(step, batch, train_x, train_y, nlp)
        
        sess.run(train_step, feed_dict={embed:x_batch, _labels:y_batch, _seqlens:seqlen_batch})

        if step % 5 == 0:
            acc, aut = sess.run([accuracy, final_output_sig], feed_dict={embed:x_batch, _labels:y_batch, _seqlens:seqlen_batch})
            
            for i in range(5):
                print(y_batch[i], aut[i])
            
            print(str(step) + '/' + str(len(train_x)) + " R0 SEQ: Accuracy at %d: %.5f" % (step, acc))
     
    # Alternating
    for step in range(0, len(train_x), batch):
        if step % 2 == 0:
            x_batch, y_batch, seqlen_batch = get_seq_batch(step, batch, train_x, train_y, nlp)
        else:
            x_batch, y_batch, seqlen_batch = get_random_batch(batch, ones_x, ones_y, nlp)
        sess.run(train_step, feed_dict={embed:x_batch, _labels:y_batch, _seqlens:seqlen_batch})
        if step % 5 == 0:
            acc, aut = sess.run([accuracy, final_output_sig], feed_dict={embed:x_batch, _labels:y_batch, _seqlens:seqlen_batch})
            
            for i in range(5):
                print(y_batch[i], aut[i])
            
            #print(str(step) + '/' + str(ran_ct) + " R" + str(e) + " RAN: Accuracy at %d: %.5f" % (step, acc))
            print(str(step) + '/' + str(len(train_x)) + " R1 TOGGLE: Accuracy at %d: %.5f" % (step, acc))

    # Alternating
    for step in range(0, len(train_x), batch):
        if step % 2 == 0:
            x_batch, y_batch, seqlen_batch = get_random_batch(batch, ones_x, ones_y, nlp)
        else:
            x_batch, y_batch, seqlen_batch = get_seq_batch(step, batch, train_x, train_y, nlp)

        sess.run(train_step, feed_dict={embed:x_batch, _labels:y_batch, _seqlens:seqlen_batch})
        if step % 5 == 0:
            acc, aut = sess.run([accuracy, final_output_sig], feed_dict={embed:x_batch, _labels:y_batch, _seqlens:seqlen_batch})
            
            for i in range(5):
                print(y_batch[i], aut[i])
            
            #print(str(step) + '/' + str(ran_ct) + " R" + str(e) + " RAN: Accuracy at %d: %.5f" % (step, acc))
            print(str(step) + '/' + str(len(train_x)) + " R2 TOGGLE: Accuracy at %d: %.5f" % (step, acc))
    

    """
    for step in range(0, len(train_x), batch):
        if step % 2 == 0:
            x_batch, y_batch, seqlen_batch = get_seq_batch(step, batch, train_x, train_y, nlp)
        else:
            x_batch, y_batch, seqlen_batch = get_random_batch(batch, train_x, train_y, nlp)
        sess.run(train_step, feed_dict={embed:x_batch, _labels:y_batch, _seqlens:seqlen_batch})
        if step % 5 == 0:
            acc, aut = sess.run([accuracy, final_output_sig], feed_dict={embed:x_batch, _labels:y_batch, _seqlens:seqlen_batch})
            
            for i in range(5):
                print(y_batch[i], aut[i])
            
            #print(str(step) + '/' + str(ran_ct) + " R" + str(e) + " RAN: Accuracy at %d: %.5f" % (step, acc))
            print(str(step) + '/' + str(len(train_x)) + " R3 TOGGLE: Accuracy at %d: %.5f" % (step, acc))
    """

    """
    for step in range(25000):
        x_batch, y_batch, seqlen_batch = get_random_batch(batch, train_x, train_y, nlp)
        sess.run(train_step, feed_dict={embed:x_batch, _labels:y_batch, _seqlens:seqlen_batch})
        if step % 5 == 0:
            acc, aut = sess.run([accuracy, final_output_sig], feed_dict={embed:x_batch, _labels:y_batch, _seqlens:seqlen_batch})
            
            for i in range(5):
                print(y_batch[i], aut[i])
            
            print(str(step) + '/' + str(ran_ct) + " R-init RAN: Accuracy at %d: %.5f" % (step, acc))
    """

    #--------------------------------------------------------------------------------------------------------------------------------
    del train_x
    del train_y
    del ones_x
    del ones_y
    tdf = pd.read_csv(os.getcwd() + "\\test.csv")
    names = list(tdf["id"])
    test_x = list(tdf["comment_text"])


    #tdft = pd.read_csv(os.getcwd() + "\\test_labels.csv")
    #test_y = list(zip(tdft["toxic"], tdft["severe_toxic"], tdft["obscene"], tdft["threat"], tdft["insult"], tdft["identity_hate"]))

    step = batch
    for i in range(0, len(test_x), step):
        print("Pred:", str(i) + '/' + str(len(test_x)))
        in_x, sq_len = get_test(test_x[i:i+step], nlp)
        output_example = sess.run(final_output_sig, feed_dict={embed:in_x, _seqlens:sq_len})
        for j in range(len(output_example)):
            buffer = [names[i+j]]
            buffer.extend(output_example[j])
            outframe.append(buffer)

    csvWriteRow(outframe, "not-fail-hopefully-lel.csv")
    saver.save(sess, os.getcwd() + '/model/test_save_lmao')


print(datetime.now() - start)

