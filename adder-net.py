


import os
import numpy as np
import pandas as pd
import tensorflow as tf 
import spacy
nlp = spacy.load('en')

print("[+] Imports Complete [+]")

what = "train"
df = pd.read_csv(os.getcwd() + "\\" + what + ".csv", nrows=5000)
#print(len(df["comment_text"]))

train_x = list(df["comment_text"])
train_y = list(zip(df["toxic"], df["insult"]))

frame = []
for i in train_x:
    buff = []
    doc = nlp(i)
    print(len(doc))
    if len(doc) > 100:
        condenser = []
        cond = np.zeros(300)
        for word in doc[0:50]:
            buff.append(word.vector)
        for word in doc[50:-50]: # could optimize no doubt
            condenser.append(word)
        for word in condenser:
            cond = np.add(cond, word.vector)
        buff.append(cond)
        for word in doc[-50:]:
            buff.append(word.vector)
    elif len(doc) < 100:
        #padding alg
    elif len(doc) = 100: 
        #rawski
        
        


    """
    vec = np.zeros(300)
    for word in doc:
        #print(len(word.vector))
        vec = np.add(vec, np.array(word.vector))
        #print(vec)
    frame.append(vec)
    """

#print("Max:", max(sizes))
#print("AVG:", int(sum(sizes)/len(sizes)))


"""
def get_batch(batch_size, data_x, data_y, nlp):
    instance_indecies = list(range(len(data_x)))
    np.random.shuffle(instance_indecies)
    batch = instance_indecies[:batch_size] 

    x1 = [data_x[i] for i in batch]
    x2 = [nlp(i) for i in x1]
    x3 = []
    for sent in x2:
        buff = []
        for word in sent:
            buff.append(word.vector)
        x3.append(buff)
    
    y = [[data_y[i]] for i in batch]

    return np.array(x3), np.array(y)
"""
"""
for i in range(20):
    print(train_y[i], train_x[i])

x, y = get_batch(batch_size, train_x, train_y, nlp)
print(x, y)
"""

"""
def RNN(x, weights, biases):
    in_size = tf.shape(x)[2]
    print(in_size)
    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, in_size])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,in_size,1)

    # 1-layer LSTM with n_hidden units.
    rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


n_hidden = 512 # number of units in RNN cell
num_classes = 2
batch_size = 2

x = tf.placeholder(tf.int32, shape=[])
y = tf.placeholder(tf.float32, shape=[]) # maybe keep batch size

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}




pred = RNN(x, weights, biases)
# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1000): #arb num of training epochs i reckon
        x_batch, y_batch = get_sentence_batch(batch_size, train_x, train_y)
        sess.run(optimizer, feed_dict={x:x_batch, y:y_batch})


        if step % 100 == 0:
            acc = sess.run(accuracy, feed_dict={_inputs:x_batch, _labels:y_batch})
            print("Accuracy at %d: %.5f" % (step, acc))
"""
