import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class SVCModel:
    def __init__(self):
        self.model = make_pipeline(
            StandardScaler(),
            SVC(
                gamma='auto',
                probability=True
            )
        )

    def fit(self, x, y):
        self.model.fit(x, y, )

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)


if __name__ == "__main__":
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    y = np.array([1, 1, 2, 2])
    model = SVCModel()
    model.fit(X, y)
    print(model.predict([[-3, -1]]))
