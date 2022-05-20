import re
import string
import tensorflow as tf

from tensorflow.keras import layers


AUTOTUNE = tf.data.AUTOTUNE
MAX_FEATURES = 10000


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')


@tf.autograph.experimental.do_not_convert
def preprocess(loader):
    sequence_length = 89

    vectorize_layer = layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=MAX_FEATURES,
        output_mode='int',
        output_sequence_length=sequence_length)

    loader_text = loader.map(lambda x, y: x)
    vectorize_layer.adapt(loader_text)

    @tf.autograph.experimental.do_not_convert
    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    loader_vec = loader.map(vectorize_text)
    return loader_vec