"""
Notes from:
https://www.tensorflow.org/programmers_guide/tensors
"""


"""
Tensor:
Generalization of vectors and matricies to potentially higher dimensions

tf.Tensor properties
    data type (float32, int32, string, etc)
    shape

Each element in the Tensor has the same data type, and the data type is always known
Shape, on the other hand, may not be fully known

Main tensors are

tf.Variable
tf.constant
tf.placeholder
tf.SparseTensor

With the exception of tf.Variable, the value of a tensor is immutable
"""

"""
--> Rank
Rank        Math entity
0           Scalar (magnitude only)
1           Vector (magnitude and direction)
2           Matrix (table of numbers)
3           3-Tensor (cube of numbers)
n           n-Tensor ("you get the idea"), "thanks, guy"

Rank 0
    mammal = tf.Variable("Elephant", tf.string)
    ignition = tf.Variable(451, tf.int16)
    floating = tf.Variable(3.14159265359, tf.float64)
    its_complicated = tf.Variable(12.3 - 3.85j, tf.complex64)

Rank 1
    mystr = tf.Variable(["Hello"], tf.string)
    cool_numbers = tf.Variable([3.14159, 2.71828], tf.float32)
    first_primes = tf.Variable([2, 3, 4, 5, 7, 11], tf.int32)

Higher Ranks
    mymat = tf.Variable([[7], [11]], tf.int16)
    myxor = tf.Variable([[False, True], [True, False]], tf.bool)
    linear_squares = tf.Variable([[4], [9], [16], [25]], tf.int32)
    squarish_squares = tf.Variable(4, 9], [16, 25]], tf.int32)
"""
"""
--> Referring to tf.Tensor slices

value = vector[1, 2] # like np indexing
"""
"""
--> Changing shape of tf.Tensor

rank_three_tensor = tf.ones([3, 4, 5])
matrix = tf.reshape(rank_three_tensor, [6, 10]) # reshaping existing content into 6x10
matrixB = tf.reshape(matric, [3, -1]) # reshape existing matrix into 3x20 (4*5). -1 tells reshape to calculate the size of this dimension
matrixAlt = tf.reshape(matrixB, [4, 3, -1]) # reshaping content into 4, 3, 5 tensor

# Note, this will throw ERROR !!!!
No possible combination of the elements can yield these dimensions
yet_Another = tf.reshape(matricAlt, [13, 2, -1])
"""

"""
--> Data types
Tensors MUST have one and only one data type
"""

"""
--> Evaluating Tensors

Once the computation graph has been built, can run computation that produces a particular tf.Tensor
ex:
constant = tf.constant([1, 2, 3])
tensor = constant * constant

print(tensor.eval()) # eval method only works when default tf.Session is active
                     # eval returns numpy array with same content as the tensor

Sommetimes, not possible to evaluate a tf.Tensor with no context, b/c value might depend on dynamic information thats not avail
For ex, tensors that depend on PLACEHOLDERs cant be evaluated without providing val for the placeholder

p = tf.placeholder(tf.float32)
t = p + 1.0
t.eval() # error, no val in placeholder
t.eval(feed_dict={p:2.0}) # succeeds b/c using feeddict to feed
"""
