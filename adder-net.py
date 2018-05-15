


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


def meaterizer(train_x, nlp):
    frame = []
    seqlens = []
    for i in train_x:
        buff = []
        doc = nlp(i)

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
            

        frame.append(buff)
    return frame, seqlens


#print("Max:", max(sizes))
#print("AVG:", int(sum(sizes)/len(sizes)))

def get_batch(batch_size, data_x, data_y, nlp):
    instance_indecies = list(range(len(data_x)))
    np.random.shuffle(instance_indecies)
    batch = instance_indecies[:batch_size] 

    x, seqlens = meaterizer([data_x[i] for i in batch], nlp)
    y = [data_y[i] for i in batch]

    return np.array(x), np.array(y), np.array(seqlens)

"""
def RNN(x, weights, biases):
    # reshape to [1, n_input]
    #x = tf.reshape(x, [-1, post_size])
    #print(x.shape)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    x = tf.reshape(x, [-1, post_size])
    print(x.shape)
    #print(x.shape)
    #print("[+] after reshape")
    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,post_size,1)
    #print(x.shape)
    #print("[+] after split")
    # 1-layer LSTM with n_hidden units.
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    #print("[+] after cell made")
    # generate prediction
    outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    #print(outputs)
    #print("--------------")
    #print(states)
    # there are n_input outputs but
    # we only want the last output

    #return tf.matmul(outputs[-1], weights['out']) + biases['out']
    softmax_w = tf.get_variable("softmax_w", [post_size, vec_size]) #[20x100]
    softmax_b = tf.get_variable("softmax_b", [vec_size]) #[1x100]
    print("S_w:", softmax_w.shape)
    print("S_b:", softmax_b.shape)
      
    return tf.matmul(outputs, softmax_w) + softmax_b
"""


"""
x, y, seqlens = get_batch(batch_size, train_x, train_y, nlp)
print(x[:2])
print("----------")
print(y[:2])
print(seqlens[:2])
"""

num_classes = 2
batch_size = 32
post_size = 128
vec_size = 300
n_hidden = 512 # number of units in RNN cell
learning_rate = .001

"""
_inputs = tf.placeholder(tf.float32, shape=[batch_size, post_size, vec_size])
_labels = tf.placeholder(tf.int32, shape=[batch_size, post_size, num_classes])
#_seqlens = tf.placeholder(tf.int32, shape=[batch_size])

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

#acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(_labels, 1), predictions=tf.argmax(_,1))

pred = RNN(_inputs, weights, biases)
# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=_labels))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
"""
"""
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1000): #arb num of training epochs i reckon
        x_batch, y_batch, seqlens = get_batch(batch_size, train_x, train_y, nlp)
        print("X Shape:", x_batch.shape)
        sess.run(optimizer, feed_dict={_inputs:x_batch, _labels:y_batch})


        if step % 100 == 0:
            _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], feed_dict={_inputs:x_batch, _labels:y_batch})
            print("Accuracy at %d: %.5f" % (step, accuracy))

    # Will need to split data into testing
    for test_batch in range(5):
        x_test, y_test, seqlen_test = get_sentence_batch(batch_size, test_x, test_y, test_seqlens)

        batch_pred, batch_acc = sess.run([tf.argmax(final_output, 1), accuracy],
                                            feed_dict={_inputs:x_test, _labels:y_test, _seqlens:seqlen_test})
        print("Test batch accuracy %d: %.5f" % (test_batch, batch_acc))
"""


def get_sentence_batch(batch_size, data_x, data_y):
    instance_indecies = list(range(len(data_x)))
    np.random.shuffle(instance_indecies)
    batch = instance_indecies[:batch_size] 

    x, seqlens = meaterizer([data_x[i] for i in batch], nlp)
    y = [data_y[i] for i in batch]

    return np.array(x), np.array(y), np.array(seqlens)


_inputs = tf.placeholder(tf.int32, shape=[batch_size, post_size, vec_size])
_labels = tf.placeholder(tf.float32, shape=[batch_size, num_classes])

embed = tf.Variable([batch_size, post_size, vec_size])

# seqlens for dynamic calculation
_seqlens = tf.placeholder(tf.int32, shape=[batch_size])

with tf.variable_scope("lstm"):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0) # Basic LSTM Cell yo
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, embed, sequence_length=_seqlens, dtype=tf.float32)

weights = {
    'linear_layer': tf.Variable(tf.truncated_normal([n_hidden, num_classes], mean=0, stddev=0.1))
}

biases = {
    'linear_layer':tf.Variable(tf.truncated_normal([num_classes], mean=0, stddev=0.1))
}

# Extract the last relevant output and use in linear layer
final_output = tf.matmul(states[1], weights["linear_layer"]) + biases["linear_layer"]
softmax = tf.nn.softmax_cross_entropy_with_logits(logits=final_output, labels=_labels)
cross_entropy = tf.reduce_mean(softmax)

train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(_labels, 1), tf.argmax(final_output, 1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))*100

#-----------------------------------------------------------------------------------------------------------------------------------
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1000): #arb num of training epochs i reckon
        x_batch, y_batch, seqlen_batch = get_sentence_batch(batch_size, train_x, train_y)
        #embed = x_batch
        print("new xbatch ------>")
        print(x_batch)
        sess.run(train_step, feed_dict={_inputs:x_batch, _labels:y_batch, _seqlens:seqlen_batch})


        if step % 100 == 0:
            acc = sess.run(accuracy, feed_dict={_inputs:x_batch, _labels:y_batch, _seqlens:seqlen_batch})
            print("Accuracy at %d: %.5f" % (step, acc))
