# LMS Risk Portal CSV Contract

## Prediction CSV

Upload a `.csv` file with the project ID column and all feature columns used by the trained LMS model.

Required shape:

```text
ID_COLUMNS + FEATURE_COLUMNS
```

## Truth CSV

Upload truth only after the prediction output exists for the same batch.

Required shape:

```text
ID_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMN
```

The target column must contain binary labels `0` or `1`.
