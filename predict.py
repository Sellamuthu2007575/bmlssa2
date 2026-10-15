"""Load the trained model and predict digits from the built-in sample set."""
import sys
import joblib
from sklearn.datasets import load_digits

def main():
    model = joblib.load("model.joblib")
    digits = load_digits()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    preds = model.predict(digits.data[:n])
    for i, p in enumerate(preds):
        print(f"sample {i}: predicted={p}  actual={digits.target[i]}")

if __name__ == "__main__":
    main()
