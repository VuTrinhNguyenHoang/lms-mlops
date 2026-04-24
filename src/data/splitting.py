from sklearn.model_selection import train_test_split
from core.config import RANDOM_STATE, TEST_SIZE

def split_train_valid(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
