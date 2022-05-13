import matplotlib.pyplot as plt
import os
import re
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

AUTOTUNE = tf.data.AUTOTUNE
MAX_FEATURES = 10000


def read_data(data_dir: str):
    batch_size = 32
    seed = 42
    train_dir = os.path.join(data_dir, "train")
    train_loader = tf.keras.utils.text_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        seed=seed
    )
    test_dir = os.path.join(data_dir, "test")
    test_loader = tf.keras.utils.text_dataset_from_directory(
        test_dir,
        batch_size=batch_size,
        seed=seed
    )
    return train_loader, test_loader


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')


@tf.autograph.experimental.do_not_convert
def process(loader):
    sequence_length = 250

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


def create_and_train_model(train_data_loader, test_data_loader):
    embedding_dim = 16
    model = tf.keras.Sequential([
        layers.Embedding(MAX_FEATURES + 1, embedding_dim),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(1)])

    model.summary()
    model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
    epochs = 10
    history = model.fit(
        train_data_loader,
        epochs=epochs)

    loss, accuracy = model.evaluate(test_data_loader)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy) #todo: loss не меняется

    history_dict = history.history
    history_dict.keys()

    acc = history_dict['binary_accuracy']
    loss = history_dict['loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    # путь до папки, в которой лежат папки train и test
    # в train и test лежит по 5 папок с текстами соответствующих категорий
    data_dir = "D://maga//2_maga_sem//nlp//net//splitted_somehow"
    train, test = read_data(data_dir)
    train_data_loader = process(train)
    test_data_loader = process(test)

    train_data_loader = train_data_loader.cache().prefetch(buffer_size=AUTOTUNE)
    test_data_loader = test_data_loader.cache().prefetch(buffer_size=AUTOTUNE)

    create_and_train_model(train_data_loader, test_data_loader)