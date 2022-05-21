import numpy as np
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
    sequence_length = 90

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


def loop_vector(vec: np.array) -> np.array:
    res = vec.copy()
    zeros_num = len(res) - np.count_nonzero(res)
    first_zero_idx = np.argmax(res == 0)
    while zeros_num > 0 and np.count_nonzero(res[first_zero_idx:]) == 0:
        nonzero_part = res[:-zeros_num]
        part_to_replace_zeros_with = res[:zeros_num]
        res = np.concatenate((nonzero_part, part_to_replace_zeros_with))
        zeros_num = len(res) - np.count_nonzero(res)
        first_zero_idx = np.argmax(res == 0)
    return res


if __name__ == "__main__":
    print(loop_vector(np.array([1, 2, 3, 4, 0, 0, 0])))