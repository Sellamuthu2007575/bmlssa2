# Digit Classifier (starter AI model)

A small neural network (MLP) that recognises handwritten digits (0-9)
using scikit-learn's built-in digits dataset.

## Setup
    pip install -r requirements.txt

## Train (creates model.joblib)
    python train.py

## Predict
    python predict.py        # first 10 samples
    python predict.py 25     # first 25 samples

## Adapting it to your own problem
- Replace `load_digits()` in train.py with your own data (X = features, y = labels).
- Swap `MLPClassifier` for another model (RandomForest, LogisticRegression, etc.).
- For regression tasks, use `MLPRegressor` and `mean_squared_error`.
