from sklearn.model_selection import train_test_split
from core.config import RANDOM_STATE, TEST_SIZE


def _can_stratify(y) -> bool:
    class_counts = y.value_counts()
    return len(class_counts) > 1 and class_counts.min() >= 2


def split_train_valid(X, y):
    stratify = y if _can_stratify(y) else None
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
