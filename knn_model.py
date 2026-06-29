from adultdata_preprocessed import (
    X_train,
    X_test,
    y_train,
    y_test
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def find_best_k(X, y, k_range=range(1, 31, 2), cv=5, plot=True):

    cv_scores = []

    for k in k_range:
        knn_temp = KNeighborsClassifier(n_neighbors=k, metric='euclidean')

        scores = cross_val_score(
            knn_temp, X, y,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )

        cv_scores.append(scores.mean())

    best_k = list(k_range)[int(np.argmax(cv_scores))]

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(list(k_range), cv_scores, marker='o', linestyle='-', color='royalblue')
        plt.axvline(
            x=best_k,
            color='red',
            linestyle='--',
            label=f'Best K = {best_k}'
        )
        plt.xlabel("Number of Neighbors (K)")
        plt.ylabel(f"Mean {cv}-Fold Cross-Validation Accuracy")
        plt.title("Selecting the Optimal K Using Cross-Validation")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return best_k, cv_scores


import time

start_time = time.time()
best_k, cv_scores = find_best_k(X_train, y_train, k_range=range(1, 31, 2), cv=5)
elapsed = time.time() - start_time

print(f"Best K selected using Cross-Validation: {best_k}")
print(f"Mean Cross-Validation Accuracy: {max(cv_scores) * 100:.2f}%")
print(f"K Search Time: {elapsed:.1f} seconds\n")

knn = KNeighborsClassifier(
    n_neighbors=best_k,
    metric='euclidean'
)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f'Accuracy: {acc * 100:.2f}%')

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()