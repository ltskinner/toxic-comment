"""
Notes taken from:
https://www.tensorflow.org/programmers_guide/variables
"""

"""
--> Variables
A TF variable is the best way to represent a shared, persistent state, that is manipulated by the program

Unlike tf.Tensor, tf.Variable exists outside of the context of a single session.run call

Internally, tf.Variable sotres a presistent tensor.
These modification are visible across multiple tf.Sessions, so multiple workers can see the same values of tf.Variable

--> Creating a Variable

my_variable = tf.get_variable("my_variable", [1, 2, 3]) # use tf.get_variable -> gives handle to Var??
                                             # makes tensor shape [1, 2, 3]
"""

"""
--> Initializing Variables
Before can be used, need to be initialized

session.run(global_variables_initializer())

To use the value of a tf.Variable, treat it like normal tf.Tensor

for reuse:

with tf.variable_scope("model"):
    output1 = my_image_filter(input1)
with tf.variable_Scope("model", reuse=True):
    output2 = my_image_filter(input2)

or

with tf.variable_Scope("model") as scope:
    output1 = my_image_filter(input1)
    scope.reuse_variables()
    output = my_image_filter(input2)
"""
