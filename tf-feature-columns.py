"""
Notes from:
https://www.tensorflow.org/get_started/feature_columns
"""

"""
Feature Columns

Think of feature columns as the intermediaries between raw data and Estimators
Feature Columns are very rich, enabling you to transform a diverse range of raw data into formats that Estimators can use
                        (Allowing for easy experimentation)


Input to a Deep Neural Network
What kind of data can a NN operate on? Numbers (tf.float32)
However, many data streams are not numbers, strings scro

ML models generally represent categorical values as simple one-hot encoded vectors

raw = {
    "tigerLength": [...],
    "tigerWidth": [...],
    "bearLength": [...],
    "bearHeight": [...]
}

Becomes
tf.feature_columns, where

feature_columns = [ numeric_column("tigerLength"),
                    numeric_column("tigerWidth"),
                    numeric_column("bearLength"),
                    numeric_column("bearHeight")
                  ]

and fed to Estimator
tf.estimator.DNNClassifier, like so

classifier = DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 10],
    n_classes=3,
    model_dir=PATH
)
"""
# Making the cols

#defaults to tf.float32
numeric_featureColumn = tf.feature_column.numeric_column(key="SepalLength")
#non default like so
numeric_feature_column = tf.feature_column.numeric_column(key="SepalLength", dtype=tf.float64)

#move to vector
vector_feature_column = tf.feature_column.numeric_column(key="Bowling", shape=10)
# 10x5
matrix_feature_column = tf.feature_column.numeric_column(key="MyMatrix", shape=[10, 5])


"""
Bucketized Column
Often, dont want to feed a number directly into a model, but instead break it into different categoris based on a range
To do so, create a "bucketized column" --> https://www.tensorflow.org/api_docs/python/tf/feature_column/bucketized_column

like year range 1980 < 1990 < 2000 < 2010
1980    [1, 0, 0, 0]
1990    [0, 1, 0, 0]
2000    [0, 0, 1, 0]
2010    [0, 0, 0, 1]

Much better than scalar b/c model can now learn 4 weights instead of 1
Also, clearly indicates the different categories, instead of allowing model to attept to make linear model w/ linear relationship
"""

# first convert the raw input into a numeric column
numeric_feature_column = tf.feature_column.numeric_column("Year")

# Then bucketize
bucketized_feature_column = tf.feature_column.bucketized_column(
    source_column=numeric_feature_column,
    boundaries=[1960, 1980, 2000])

