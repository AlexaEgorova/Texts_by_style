import os
import tensorflow as tf


TRAIN = "train"
VAL = "val"
TEST = "test"
BATCH_SIZE = 32
SEED = 42

class DataReader():
    @staticmethod
    def read(data_dir: str):
        """data_dir - путь до директории с папками train, validation, test."""
        labels = os.listdir(os.path.join(data_dir, TRAIN))
        subfolders = [TRAIN, VAL, TEST]
        tensors = []
        for subfolder_name in subfolders:
            subfolder_path = os.path.join(data_dir, subfolder_name)
            tensor = tf.keras.utils.text_dataset_from_directory(
                subfolder_path,
                batch_size=BATCH_SIZE,
                seed=SEED
            )
            tensors.append(tensor)
        return tensors, labels


if __name__ == "__main__":
    path = r"D://maga//2_maga_sem//nlp//net//splitted_somehow"
    tensors, labels = DataReader().read(path)
    train, validation, test = tensors
    print(labels)