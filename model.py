import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from colorama import init, Fore, Style
init(autoreset=True)

os.makedirs("outputs", exist_ok=True)

DATASET = "dataset.csv"

FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember"
]

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def load_data():
    df = pd.read_csv(DATASET)
    df["BalanceZero"] = (df["Balance"] == 0).astype(int)

    X = df[FEATURES + ["BalanceZero"]].values.astype(float)
    y = df["Exited"].values.astype(int)
    return X, y


def split(X, y, test_size=0.2):
    idx = np.random.permutation(len(y))
    cut = int(len(y) * (1 - test_size))
    return X[idx[:cut]], X[idx[cut:]], y[idx[:cut]], y[idx[cut:]]


def standardize(X, mu, sigma):
    return (X - mu) / (sigma + 1e-8)


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def get_metrics(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    acc = (TP + TN) / (TP + TN + FP + FN + 1e-9)
    prec = TP / (TP + FP + 1e-9)
    rec = TP / (TP + FN + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)

    return acc, prec, rec, f1, np.array([[TN, FP], [FN, TP]])


# ─────────────────────────────────────────────
# MODEL 1: LOGISTIC REGRESSION
# ─────────────────────────────────────────────

class LogisticRegression:
    def __init__(self, lr=0.1, epochs=300):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0

        for _ in range(self.epochs):
            z = X @ self.w + self.b
            p = self.sigmoid(z)

            dw = (X.T @ (p - y)) / n
            db = np.mean(p - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self, X):
        return self.sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


# ─────────────────────────────────────────────
# MODEL 2: RANDOM FOREST
# ─────────────────────────────────────────────

class DecisionTree:
    def fit(self, X, y):
        self.feature = np.random.randint(X.shape[1])
        self.threshold = np.mean(X[:, self.feature])

    def predict(self, X):
        return (X[:, self.feature] > self.threshold).astype(int)


class RandomForest:
    def __init__(self, n_trees=20):
        self.n_trees = n_trees

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            idx = np.random.choice(len(y), len(y), replace=True)
            tree = DecisionTree()
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)

    def predict_proba(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(preds, axis=0)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────

def plot_confusion(cm_list, names):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, cm, name in zip(axes, cm_list, names):
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()


def plot_metrics(results):
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(results))

    for i, label in enumerate(labels):
        values = [r[i] for r in results]
        plt.bar(x + i*0.2, values, width=0.2, label=label)

    plt.xticks(x + 0.3, ["Logistic", "RandomForest"])
    plt.legend()
    plt.savefig("outputs/model_comparison.png")
    plt.close()


# ─────────────────────────────────────────────
# PREDICTION (WITH RANGES)
# ─────────────────────────────────────────────

def predict_customer(models, mu, sigma):

    print(Fore.CYAN + "\n📥 Enter Customer Details (use given ranges)\n" + Style.RESET_ALL)

    CreditScore = int(input("Credit Score (300 - 900): "))
    Age = int(input("Age (18 - 100): "))
    Tenure = int(input("Tenure in years (0 - 10): "))
    Balance = float(input("Balance (0 - 2,000,000): "))
    NumOfProducts = int(input("Number of Products (1 - 4): "))
    HasCrCard = int(input("Has Credit Card (1/0): "))
    IsActiveMember = int(input("Is Active Member (1/0): "))

    BalanceZero = 1 if Balance == 0 else 0

    X = np.array([[CreditScore, Age, Tenure, Balance,
                   NumOfProducts, HasCrCard, IsActiveMember, BalanceZero]])

    X = standardize(X, mu, sigma)

    print(Fore.MAGENTA + "\n📊 Churn Risk Prediction:\n" + Style.RESET_ALL)

    for name, model in models:
        proba = model.predict_proba(X)[0]

        if proba >= 0.6:
            print(Fore.RED + f"{name}: 🔴 HIGH RISK ({proba*100:.1f}%)")
        elif proba >= 0.4:
            print(Fore.YELLOW + f"{name}: 🟡 MEDIUM RISK ({proba*100:.1f}%)")
        else:
            print(Fore.GREEN + f"{name}: 🟢 LOW RISK ({proba*100:.1f}%)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print(Fore.CYAN + "🚀 Training models...\n" + Style.RESET_ALL)

    X, y = load_data()
    X_tr, X_te, y_tr, y_te = split(X, y)

    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0)

    X_tr = standardize(X_tr, mu, sigma)
    X_te = standardize(X_te, mu, sigma)

    models = [
        ("Logistic", LogisticRegression()),
        ("RandomForest", RandomForest())
    ]

    results = []
    cms = []
    trained_models = []

    for name, model in models:
        print(Fore.YELLOW + f"🔧 Training {name}..." + Style.RESET_ALL)
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_te)
        acc, prec, rec, f1, cm = get_metrics(y_te, y_pred)

        print(Fore.GREEN + f"{name} → Acc:{acc:.3f}  F1:{f1:.3f}" + Style.RESET_ALL)

        results.append([acc, prec, rec, f1])
        cms.append(cm)
        trained_models.append((name, model))

    predict_customer(trained_models, mu, sigma)

    print(Fore.MAGENTA + "\n📊 Saving graphs..." + Style.RESET_ALL)
    plot_confusion(cms, [m[0] for m in models])
    plot_metrics(results)

    print(Fore.GREEN + "\n✅ Done! Check 'outputs' folder.\n" + Style.RESET_ALL)


if __name__ == "__main__":
    main()


# Credit Score: 750
# Age: 35
# Tenure: 6
# Balance: 500000
# Number of Products: 2
# Has Credit Card (1/0): 1
# Is Active Member (1/0): 1

# Credit Score: 350
# Age: 60
# Tenure: 1
# Balance: 0
# Number of Products: 3
# Has Credit Card (1/0): 0
# Is Active Member (1/0): 0