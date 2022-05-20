import numpy as np
import tensorflow as tf


from sklearn.metrics import f1_score, log_loss


from reader import DataReader
from processing import preprocess
from model import SVCModel


AUTOTUNE = tf.data.AUTOTUNE
MAX_FEATURES = 10000


if __name__ == "__main__":
    path = r"D://maga//2_maga_sem//nlp//net//dataset"
    tensors, class_names = DataReader().read(path)
    train_tensor, validation_tensor, test_tensor = tensors

    train_vec = preprocess(train_tensor)
    validation_vec = preprocess(validation_tensor)
    test_vec = preprocess(test_tensor)

    train = next(iter(train_vec))
    validation = next(iter(validation_vec))
    test = next(iter(test_vec))

    train_data = train[0].numpy()
    train_labels = train[1].numpy()

    model = SVCModel()
    model.fit(train_data, train_labels)

    test_data = test[0].numpy()
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

        # zeros_num = len(example) - np.count_nonzero(example)
        # print("zeros num:", zeros_num)
        # print("example:", example)
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
