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
"""
# first convert the raw input into a numeric column
numeric_feature_column = tf.feature_column.numeric_column("Year")

# Then bucketize
bucketized_feature_column = tf.feature_column.bucketized_column(
    source_column=numeric_feature_column,
    boundaries=[1960, 1980, 2000])
"""
"""
Categorical Identity Column
Special cases of bucketiszed items
0 [1, 0, 0, 0] where 0 is representative of "kitchenware
1 [0, 1, 0, 0]  1="electronics"
2 [0, 0, 1, 0]  2="sport" 
3 [0, 0, 0, 1]  3="living"
"""
"""
# to do this, use tf.feature_column.categorical_column_with_identity
identity_feature_column = tf.feature_column.categorical_column_with_identity(
    key='my_feature_b',
    num_buckets=4) # values [0, 4)

# In order for the preceding call to work, the input_fn() must return a DICTIONARY containing  'my_feature_b' as a key
# Furthermore, the values assigned to 'my_feature_b' must belong to the set [0, 4)

def input_fn():
    return({'my_feature_a':[7, 9, 5, 2], 'my_feature_b':[3, 1, 2, 2], [Label_values]}) # note really sure what Label_values are...
"""

"""
Categorical Vocabulay Column
We cannot input strings directly into a model. Instead, we must first map strings to numeric or categorical values
Categoricalvocab columns provied a good way to represent strings as a one hot encoded vector

"kitchenware" --> [1, 0, 0]
"electronics" --> [0, 1, 0]
"sports"      --> [0, 0, 1]

two ways to do this:
    1) tf.feature_column.categorical_column_with_vocabulary_list # LIST
    2) tf.feature_column.categorical_column_With_vocabualry_file # FILE

# Given input "feature_name_from_input_fn" which is a string, create categorical feature by mapping th einput to one of the elements in list
vocalulary_feature_column = tf.feature_column.categorical_column_with_vocabualry_list(
    key=feature_name_from_input_fn,
    vocabulary_list=["kitchenware", "electronics", "sports"])

# Primary drawback of above is needing to type out vocab_list components (meh, dynamic???), so can do file of premade
vocabulary_Feature_column = tf.feature_column.categorical_column_with_vocabulary_file(
    key=feature_name_from_input_fn,
    vocabulary_file="product_class.txt",
    vocabulary_size=3)

where product_class.txt -->
kitchenware
electronics
sports
"""

"""
Hashed Column
ore often than not, the number of categories is so big its not possible to have indiv cats for each vocab word or integer
    (b/c consume too much memory)
Instead, we can ask "how many categories am I willing to have for my input?"

tf.feature_column.categorical_column_with_hash_bucket # allows you to specify eh
-->
hashed_feature_column = tf.feature_column.categorical_column_with_hash_bucket(
    key="some_feature",
    hash_buckets_size=100)

"As with many counterintuitive phenomena in machine learning,
it turns out that hashing often works well inpractive.
Thats because hash categories provide the model with some separation. 
The model can use additional features to further separate kitchenware from sports."
"""

"""
Crossed column
Combining features into a single feature, better known as a "feature crosses",
    enables model to learn separate weights for each combination of features

Suppose, want to calc the real estate prices in Atl, GA.
Suppose, ATL is represented in 100x100 grid of rectangualr sections, iden 10,000 sections by a SINGLE feature of lat x long
this creates a much stronger signal than lat or long alone
"""
#here, use combination of
#    bucketized_column
#    tf.feature_column.crossed_column

def make_dataset(latitude, longitude, labels):
    assert latitude.shape == longitude.shape == labels.shape # lmehoh what eez thees?!

    features = { 'latitude': latitude.flatten(), #what eez thees?? flatten removes dimensionality, keeps copy of original
                 'longitude': longitude.flatten()}
    labels=labels.flatten()

    return tf.data.Dataset.from_tensor_slices((features, labels))

# bucketize the latitude and longitude using the 'edges'
latitude_bucket_fc = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column('latitude'),
    list(atlanta.latitude.edges)))

longitude_bucket_fc = tf.feature_column.bucktized_column(
    tf.feature_column.numeric_column('longitude'),
    list(Atlanta.longitude.edges))

# cross the bucketized columns, using 5000 hash bins
crossed_lat_lon_fc = tf.feature_column.crossed_column(
    [latitude_bucket_fc, longitude_bucket_fc], 5000)

fc = [
    latitude_bucket_fc,
    longitude_bucket_fc,
    crossed_lat_lon_fc]

# build and train the Estimator
est = tf.estimator.LinearRegressor(rc, ...)

"""
feature crosses can be made from either of the following:
    feature names, names from the dict returned from input_fn
    any categorical column, except categorical_column_with_hash_bucket (since crossed_column hashes the input)

whe feature columns latitude_bucket_fc and longitude_bucket_fc are crossed,
TensorFlow creates:
(latitude_fc, longitude_fc) pairs for each example
(0, 0), (0, 1) .... (0, 99)
(1, 0)
(99, 0) (99, 1) ... (99, 99)

!!!! Also, still good idea to pass uncrossed original values to column so model can distinguish hash collision
"""

"""
Indicator and Embedding Columns
Indicator columns and embedding columns never work on features directly, but instead take categorical cols as input
!!!! Yooo these are the embedding bois that the vector key dicts use !!!

When using an indicator column, were telling TF to do exactly what weve seen in the categorical product_class ex.
That is, an indicator column trats each category as an element in a one-hot encoded vector

categorical column = ... # lmao what??? no example
# represent cat col as ind col
indicator_column = tf.feature_column.indicator_column(categorical_column)

# one hot encoding all the way out is not scalable (think million different categories)
Embedding column fixes this by representing the data as a lower dimensional, oridnay vector
    in which each cell can contain any number, not just 0 or 1. By permitting a richer palette
    of numbers for every cell, an embedding contains far fewer cells than an indicator column


Compareing indicator and embedding cols
suppose input examples consist of different words from a limited palette of only 81 words
further, suppose that the data set provides the following input words in 4 separate examples
    "dog" -> 0
    "spoon" -> 32
    "scissors" -> 79
    "guitar" -> 80

categorical features via:
    cat_col_w_hash_bucket
    cat_col_w_vocabulary_list
    cat_col_w_vocabulary_file

            [0, 3]                          [0, 81]
        Lookup Table                 Indicator column
# 0 [.324, .453, .838]          [1, ..., 0, ..., 0, 0]
#32 [.543, .904, .874]          [0, ..., 1, ..., 0, 0]
#79 [.294, .112, .578]          [0, ..., 0, ..., 1, 0]
#80 [.722, .905, .987]          [0, ..., 0, ..., 0, 0]

Embedding column stores categforical data in lower dim vector
these numbers are random, but training determines the actual numbers

When an example is processed, one of the cat_cols_w... function maps the example string to a
numerical categorical value.

How do the vals in the embedding vectors magically get assigned? --> via training
Embedding columns ioncreases model capabilities, since embedding vectors learns new relationships

embedding dimensions = number_of_categories**.25
3 = 81**.25
"""
"""
categorical_column = ["hello", "end", "of", "the", "page"]
# Represent the categorical column as an embedding column
# This means creating a one hot vector with one element for each category
embedding_column = tf.feature_column.embedding_column(
    categorical_column=categorical_column,
    dimension=dimension_of_embedding_vector)
"""

"""
--> Passing Feature Columns to Estimators
Not all estimators permit all types of feature_columns arguments
    Linear Classifier and Linear Regressor: Accepts all types!!
    DNNClassifier and DNNRegressor: only accepts dense columns
        Other column types must be wrapped in either an indicator_column or embedding_column
    DNNLinearCombinedClassifier and DNNLinearCombinedRegressor:
        linear_feature_columns argument accepts any feature column type
        dnn_feature_columns arg only accepts dense columns
"""






