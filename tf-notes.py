"""
Notes taken from:
https://www.tensorflow.org/programmers_guide/low_level_intro
"""


import numpy as np 
import tensorflow as tf 

sess = tf.Session()

"""

3           rank 0 tensor, scalar with shape []
[1, 2, 3]   rank 1 tensor, vector with shape [3]
[[1, 2, 3,], [4, 5, 6]] rank 2 tensor, matrix with shape [2, 3]
[[[1, 2, 3]], [[7, 8, 9]]] # rank 3 tensor with shape [2, 1, 3] 

TensorFlow uses numpy arrays to represent tensor values
-----------------------------------------------------------------

TensorFlow Core Walkthrough

1) Building the Computational Graph, with tf.Graph
2) Running the Computational Graph, using tf.Session

--> Graph
A computational graph is a series of TF operations arranged into a graph. The graph is composed of two objects:
    1) Operations (ops): the nodes of the graph. Operations are calculations that consume and produce tensors
    2) Tensors: The edges in the graph. These are the values that flow through the graph. Tensors are just handles

--> Session
To evaluate tensors, instantiate a tf.Session object, aka session. Sessions encapsulate the state of the
    TF runtime, and run TF operations.

    tf.Graph is like .py
    tf.Sesison is like python

#               val
a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) #tf.float32 implicitly
total = a + b 
print(a)
print(b)
print(total)
#Tensor("Const:0", shape=(), dtype=float32)
#Tensor("Const_1:0", shape=(), dtype=float32)
#Tensor("add:0", shape=(), dtype=float32)

sess = tf.Session()
print(sess.run(total))
# 7.0
"""
"""
sess = tf.Session()
vec = tf.random_uniform(shape=(3,))
out1 = vec + 1
out2 = vec + 2
print(sess.run(vec)) # [ 0.15037215  0.17862737  0.86591649]
print(sess.run(vec)) # [ 0.90985668  0.34185338  0.49050343] --> new exe, new randoms
print(sess.run((out1, out2))) # (array([ 1.02142251,  1.29110551,  1.87209904], dtype=float32), array([ 2.02142239,  2.29110551,  2.87209892], dtype=float32))
"""

""" 
--> Feeding
Graph can(should) be parameterized to accept external inputs, known as placeholders.
A placeholder is a promise to provide a value later, like a function argument
LIKE A FUNCTION ARGUMENT
"""
"""
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y

sess = tf.Session()
print(sess.run(z, feed_dict={x: 3, y: 4.5}))
print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))
# 7.5
# [ 3.  7.] # so basically, the value of the placeholder doesnt really matter as long as the operations
#             done on them have aimeable dimensions? maybe idk lmao fuck
"""

"""
Datasets

Placeholders work for simple experiments, but Datasets are the preferred method of streaming data to a model

To get ar unable tf.Tensor from a dataset:
    1) you must first confert it to a tf.data.Iterator
    2) then call the iterators .get_next() method

Simplest way to create an Iterator is with the make_one_shot_iterator method
Following code the 'next_item' tensor will return a row from the my_data array on each run call:
"""
"""
my_data = [[0, 1], [2, 3], [4, 5], [6, 7]]
slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()

sess = tf.Session()

while True:
    try:
        print(sess.run(next_item))
    except tf.errors.OutOfRangeError:
        break
"""
"""
r = tf.random_normal([10, 3])
dataset = tf.data.Dataset.from_tensor_slices(r)
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()

sess.run(iterator.initializer)
while True:
    try:
        print(sess.run(next_row))
    except tf.errors.OutOfRangeError:
        break
"""

#---------------------------------------------------------------------------------------------------------

"""
--> Layers
A trainable model must modify the values in the graph to get new outputs with the same input. 
Layers are the preferred way to add trainable parameters to a graph

Layers package together both:
    the variables
    the operations that act upon the variables

Ex. densly-connected layer performs a weighted sum across all inputs for each output
    and applies an optional activation function
    - Connection weights and biases are managed by the LAYER OBJECT


--> Creating layers
The following code creates a Dense layer that takes a batch of input vectors
    and produces a single output value for each. 
    To apply a layer ot an input, call the layer as if it were a function
"""
"""
x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1) # do believe units=1 is number of outputs
y= linear_model(x)
"""
"""
The layer inspects its input to determine sizes for its internal variables
This is why the shape of 'x' is set as placeholder, so layer can build weight matrix of correct size

the CALCULATION of the output of y has been defined
    but first, before the calculation can be run, the layer(s) need to be initialized

--> Initializing Layers
The layer contains variable that must be initialized before they can be used.
It is possible to initialize variables individially, but...
    you can easily initialize all variables in a TensorFlow graph as follows:
"""
"""
init = tf.global_variables_initializer()
sess.run(init)
"""
"""
--> Executing Layers
Now that the layer is initialized, we can evaluate the 'linear_model' otuput tensor as we would any other tensor
"""
"""
print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))
#[[0.66208327]
# [2.6112342 ]]
"""
"""
--> Layer Function shortcuts
For each layer class (like tf.layers.Dense) TensorFlow also supplies a SHORTCUT FUNCTION
    (like tf.layers.dense) (note .Dense vs .dense)
    The only difference is that the shortcut function versions create and run the layer in a single call
Following code is equivalent to the above sequence


x = tf.placeholder(tf.float32, shape=[None, 3]) #same)
y = tf.layers.dense(x, units=1) #removes placeholder 'linear_model'

init = tf.global_variables_intitializer()
sess.run(init)

print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))
"""

#-----------------------------------------------------------------------------------------------------------------------

"""
--> Feature columns
The easiest way to experiement with feature columns is using the:
    $ tf.feature_column.input_layer function
    This function only accepts DENSE COLUMNS as inputs (https://www.tensorflow.org/get_started/feature_columns)
    so to view the result of a cetegorical column, it must bewrapped in 
    $ tf.feature_column.indicator_column
"""
"""
features = {
    'sales': [[5], [10], [8], [9]],
    'department': ['sports', 'sports', 'gardening', 'gardening']
}

department_column = tf.feature_column.categorical_column_with_vocabulary_list(
    'department', ['sports', 'gardening']
)
department_column = tf.feature_column.indicator_column(department_column)

columns = [
    tf.feature_column.numeric_column('sales'),
    department_column
]

inputs = tf.feature_column.input_layer(features, columns)

# running the inputs tensor will parse the features into a batch of vectors
# Feature columns can have internal state, like layers, so they often need ot be initialized
# Categorical columns use 'lookup tables' internally and these require a separate intialization op
#       tf.tables_initializer     https://www.tensorflow.org/api_docs/python/tf/contrib/lookup

var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()
sess = tf.Session()
sess.run((var_init, table_init))

print(sess.run(inputs))

# [hot, hot, sales] 
#[[ 1.  0.  5.]
# [ 1.  0. 10.]
# [ 0.  1.  8.]
# [ 0.  1.  9.]]
"""
#-----------------------------------------------------------------------------------------------------
#--------------------------------------- T*R*A*I*N*I*N*G ---------------------------------------------
#-----------------------------------------------------------------------------------------------------

# 1) Define the data
x = tf.constant([[1], [2], [3],[4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

# 2) Define the model
#       a simple linear model, w/ one output (units=1)
linear_model = tf.layers.Dense(units=1)
y_pred = linear_model(x)

# 3) Evaluate the predictions
#sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y_pred))
#[[0.53115714]
# [1.0623143 ]
# [1.5934714 ]
# [2.1246285 ]]

# Note, the model hadnt been trained so the four "predicted" values arent very good

# ------------------- LOSS ----------------------------
# b/c of this, we need to optimize the model
# this is done by defininf the Loss
# in this case, using mean square error, std loss for regression models
# *note* possible to do manually with lower level math operations, but tf.losses() works well

loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
print(sess.run(loss))
# 6.6350455

#-------------------------------------- TRAINING ----------------------------------

# Utilizes "optimizers" - https://developers.google.com/machine-learning/glossary/#optimizer
# subclasses of tf.train.Optimizer
#       These incrementally change each variable in order to minimize the loss
#       Simplest optimization alg is gradient descent, tf.train.GradientDescentOptimizer

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

for i in range(1000):
    _, loss_value = sess.run((train, loss))
    
    if i % 10 == 0:
        print(i, loss_value)

print(sess.run(y_pred))
"""
range(100)
[[-0.37176386]
 [-1.1801451 ]
 [-1.9885263 ]
 [-2.7969077 ]]
"""
"""
range(1000)
[[-0.04365891]
 [-1.0211558 ]
 [-1.9986527 ]
 [-2.9761496 ]]
"""
"""
actual
[[0], [-1], [-2], [-3]]
"""

""" 
Definitions:
--> https://developers.google.com/machine-learning/glossary/#optimizer
--> https://www.tensorflow.org/api_docs/python/tf/contrib/lookup
--> https://www.tensorflow.org/get_started/feature_columns


Next:
--> https://www.tensorflow.org/get_started/custom_estimators
--> https://www.tensorflow.org/programmers_guide/graphs
--> https://www.tensorflow.org/programmers_guide/tensors
--> https://www.tensorflow.org/programmers_guide/variables

--> https://www.tensorflow.org/programmers_guide/
--> https://www.tensorflow.org/tutorials/
"""
