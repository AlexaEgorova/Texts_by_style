import os
import numpy as np
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import seaborn as sns
from tensorflow.python.data import Dataset
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Lambda

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])


def get_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


DATA_DIR = 'dataset_stemming'
SEED = 42
BATCH_SIZE = 32
SET_NAMES = ['test', 'train', 'val']
VOCAB_SIZE = 1000

sets = {}
for ds in SET_NAMES:
    directory = os.path.join(DATA_DIR, ds)
    sets[ds] = tf.keras.utils.text_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='categorical',
        class_names=['conversation', 'fiction', 'formal', 'journalistic', 'science'],
        batch_size=BATCH_SIZE,
        max_length=None,
        shuffle=True,
        seed=SEED,
        validation_split=None,
        subset=None,
        follow_links=False
    )
'''
    if ds != 'val':
        print('init')
        print(sets[ds])
        features = tf.data.Dataset.from_tensor_slices([tf.strings.join([mset[0]]*5) for mset in sets[ds]])
        labels = tf.data.Dataset.from_tensor_slices([mset[1] for mset in sets[ds]])
        sets[ds] = Dataset.zip((features, labels))
        print('changed')
        print(sets[ds])
    

sets['train'] = sets['train'].batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
sets['test'] = sets['test'].batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

'''

encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(sets['train'].map(lambda text, label: text))

model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, kernel_regularizer=regularizers.l2(0.001), dropout=0.3, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, kernel_regularizer=regularizers.l2(0.001), dropout=0.3)),
    tf.keras.layers.Dense(5, kernel_regularizer=regularizers.l2(0.001), activation='softmax')
])

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=[get_f1])

history = model.fit(sets['train'], epochs=150,
                    validation_data=sets['test'],
                    validation_steps=6)

test_loss, test_acc = model.evaluate(sets['val'])

print('Test Loss:', test_loss)
print('Test f1:', test_acc)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history, 'get_f1')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.ylim(0, None)

plt.show()

y_pred = model.predict(sets['val'])
predicted_categories = tf.argmax(y_pred, axis=1)
true_categories = tf.concat([tf.argmax(y, axis=1) for x, y in sets['val']], axis=0)
cm = confusion_matrix(predicted_categories, true_categories)
print(cm)

ax = sns.heatmap(cm, annot=True, cmap='Blues')

ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

ax.xaxis.set_ticklabels(['conversation', 'fiction', 'formal', 'journalistic', 'science'])
ax.yaxis.set_ticklabels(['conversation', 'fiction', 'formal', 'journalistic', 'science'])

plt.show()
