import numpy as np
import tensorflow as tf

from sklearn.metrics import f1_score, log_loss

from reader import DataReader
from processing import preprocess, loop_vector
from svc_model import SVCModel


AUTOTUNE = tf.data.AUTOTUNE
MAX_FEATURES = 10000


def run_svc():
    path = r"D://maga//2_maga_sem//nlp//net//dataset"
    tensors, class_names = DataReader().read(path)
    train_tensor, _, test_tensor = tensors

    train_vec = preprocess(train_tensor)
    test_vec = preprocess(test_tensor)

    train = next(iter(train_vec))
    test = next(iter(test_vec))

    train_data = train[0].numpy()
    train_list = []
    for example in train_data:
        train_list.append(loop_vector(example).tolist())
    train_data = np.array(train_list)
    train_labels = train[1].numpy()

    model = SVCModel()
    model.fit(train_data, train_labels)

    test_data = test[0].numpy()
    test_list = []
    for example in test_data:
        test_list.append(loop_vector(example).tolist())
    test_data = np.array(test_list)
    test_labels = test[1].numpy()

    y_pred = []
    y_true = []
    y_prob = []

    for example, answer in zip(test_data, test_labels):
        predicted_proba = model.predict_proba([example])
        predicted_class = model.predict([example])[0]
        y_pred.append(predicted_class)
        y_true.append(answer)
        y_prob.append(predicted_proba.tolist()[0])
        predicted_class = class_names[predicted_class]

        zeros_num = len(example) - np.count_nonzero(example)
        print("zeros num:", zeros_num)
        print("example:", example)
        print("label:", class_names[answer])
        print("proba:", predicted_proba[0])
        print("prediction:", predicted_class)
        # print("example_len:", np.count_nonzero(example))
        print()

    print()
    score = f1_score(y_true, y_pred, average='micro')
    print("f1-score:", score)
    logloss = log_loss(y_true, y_prob)
    print("log_loss:", logloss)

if __name__ == "__main__":
    run_svc()
