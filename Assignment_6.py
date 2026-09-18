"""
23CSE301 - Lab Session 06
kNN: Own Implementation (Version 1) vs Scikit-learn (Version 2) vs GenAI-developed (Version 3)

Dataset: sklearn Digits dataset, classes 0 and 1 (confirmed as project data).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import unittest
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ============================================================
# DATA PREPARATION
# ============================================================

def load_project_data(selected_classes=(0, 1)):
    """Loads Digits dataset, reduced to two classes (0 and 1) - confirmed project data."""
    digits = load_digits()
    X = digits.data
    y = digits.target
    mask = np.isin(y, selected_classes)
    return X[mask], y[mask]


# ============================================================
# LAB 05 / VERSION 1 - MY CUSTOM KNN (UNCHANGED FROM LAB 05)
# ============================================================

# A1(a) ENCODING
def encode_data(X):
    return np.asarray(X, dtype=float)


def calculate_mean(values):
    valid_values = [value for value in values if not np.isnan(value)]
    if len(valid_values) == 0:
        return 0
    return sum(valid_values) / len(valid_values)


def calculate_median(values):
    valid_values = [value for value in values if not np.isnan(value)]
    if len(valid_values) == 0:
        return 0
    valid_values = sorted(valid_values)
    n = len(valid_values)
    if n % 2 == 1:
        return valid_values[n // 2]
    return (valid_values[n // 2 - 1] + valid_values[n // 2]) / 2


def calculate_mode(values):
    valid_values = [value for value in values if not np.isnan(value)]
    if len(valid_values) == 0:
        return 0
    frequency = {}
    for value in valid_values:
        frequency[value] = frequency.get(value, 0) + 1
    maximum_frequency = max(frequency.values())
    mode_values = [value for value in frequency if frequency[value] == maximum_frequency]
    return min(mode_values)


def impute_missing_values(data, method="mean"):
    data = np.asarray(data, dtype=float).copy()
    for column in range(data.shape[1]):
        column_values = data[:, column]
        if np.isnan(column_values).any():
            if method == "mean":
                replacement = calculate_mean(column_values)
            elif method == "median":
                replacement = calculate_median(column_values)
            elif method == "mode":
                replacement = calculate_mode(column_values)
            else:
                raise ValueError("Method must be mean, median or mode.")
            for row in range(data.shape[0]):
                if np.isnan(data[row, column]):
                    data[row, column] = replacement
    return data


# A1(c) DISTANCE CALCULATION
def euclidean_distance(point1, point2):
    total = 0
    for i in range(len(point1)):
        total += (point1[i] - point2[i]) ** 2
    return np.sqrt(total)


# A1(d) SORTING ALGORITHMS
def bubble_sort(items):
    items = items.copy()
    n = len(items)
    for i in range(n):
        for j in range(0, n - i - 1):
            if items[j][0] > items[j + 1][0]:
                items[j], items[j + 1] = (items[j + 1], items[j])
    return items


def selection_sort(items):
    items = items.copy()
    n = len(items)
    for i in range(n):
        minimum_index = i
        for j in range(i + 1, n):
            if (items[j][0] < items[minimum_index][0] or
                    (items[j][0] == items[minimum_index][0] and items[j][1] < items[minimum_index][1])):
                minimum_index = j
        items[i], items[minimum_index] = (items[minimum_index], items[i])
    return items


def insertion_sort(items):
    items = items.copy()
    for i in range(1, len(items)):
        current = items[i]
        j = i - 1
        while j >= 0:
            if (items[j][0] > current[0] or
                    (items[j][0] == current[0] and items[j][1] > current[1])):
                items[j + 1] = items[j]
                j -= 1
            else:
                break
        items[j + 1] = current
    return items


def sort_distances(items, algorithm="selection"):
    if algorithm == "bubble":
        return bubble_sort(items)
    elif algorithm == "selection":
        return selection_sort(items)
    elif algorithm == "insertion":
        return insertion_sort(items)
    else:
        raise ValueError("Choose bubble, selection or insertion.")


# A1(e) IDENTIFY K NEAREST NEIGHBORS
def identify_neighbors(distances, training_labels, k, algorithm="selection"):
    items = []
    for index in range(len(distances)):
        items.append((distances[index], index, training_labels[index]))
    sorted_items = sort_distances(items, algorithm)
    return sorted_items[:k]


# A1(f) CLASS EVALUATION AND ASSIGNMENT
def majority_vote(neighbors):
    class_counts = {}
    for neighbor in neighbors:
        label = neighbor[2]
        class_counts[label] = (class_counts.get(label, 0) + 1)
    maximum_count = max(class_counts.values())
    candidate_classes = [label for label in class_counts if class_counts[label] == maximum_count]
    for neighbor in neighbors:
        if neighbor[2] in candidate_classes:
            return neighbor[2]
    return candidate_classes[0]


# A2 (Lab 05) - WEIGHTED KNN
def weighted_vote(neighbors):
    class_weights = {}
    epsilon = 1e-10
    for neighbor in neighbors:
        distance = neighbor[0]
        label = neighbor[2]
        weight = 1 / (distance + epsilon)
        class_weights[label] = (class_weights.get(label, 0) + weight)
    maximum_weight = max(class_weights.values())
    candidate_classes = [label for label in class_weights if class_weights[label] == maximum_weight]
    for neighbor in neighbors:
        if neighbor[2] in candidate_classes:
            return neighbor[2]
    return candidate_classes[0]


def weighted_knn_predict(X_train, y_train, X_test, k=3, algorithm="selection"):
    predictions = []
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train)
    for test_point in X_test:
        distances = np.sqrt(np.sum((X_train - test_point) ** 2, axis=1))
        neighbors = identify_neighbors(distances, y_train, k, algorithm)
        predicted_class = weighted_vote(neighbors)
        predictions.append(predicted_class)
    return np.array(predictions)


class WeightedKNN:
    def __init__(self, k=3, algorithm="selection"):
        self.k = k
        self.algorithm = algorithm
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)

    def predict(self, X):
        if self.X_train is None:
            raise ValueError("Model must be fitted before prediction.")
        return weighted_knn_predict(self.X_train, self.y_train, X, self.k, self.algorithm)

    def score(self, X, y):
        predictions = self.predict(X)
        y = np.asarray(y)
        correct = np.sum(predictions == y)
        return correct / len(y)


# VERSION 1 - MY CUSTOM KNN (Lab 05 A7)
def custom_knn_predict(X_train, y_train, X_test, k=3, algorithm="selection"):
    predictions = []
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    for test_point in X_test:
        distances = np.sqrt(np.sum((X_train - test_point) ** 2, axis=1))
        neighbors = identify_neighbors(distances, y_train, k, algorithm)
        predicted_class = majority_vote(neighbors)
        predictions.append(predicted_class)
    return np.array(predictions)


class CustomKNN:
    """Version 1 - KNN algorithm with code written by me (Lab 05)."""

    def __init__(self, k=3, algorithm="selection"):
        self.k = k
        self.algorithm = algorithm
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)

    def predict(self, X):
        if self.X_train is None:
            raise ValueError("Model must be fitted before prediction.")
        return custom_knn_predict(self.X_train, self.y_train, X, self.k, self.algorithm)

    def score(self, X, y):
        predictions = self.predict(X)
        y = np.asarray(y)
        correct = np.sum(predictions == y)
        return correct / len(y)


# ============================================================
# VERSION 2 - SCIKIT-LEARN KNN
# ============================================================

def make_sklearn_knn(k=3):
    """Factory for Version 2 model, kept modular for the comparison harness."""
    return KNeighborsClassifier(n_neighbors=k)


# ============================================================
# LAB 06 / VERSION 3 - GENAI KNN
# GenAI Tool Used: Claude
# Independent implementation: vectorized distance computation and
# np.argpartition-based neighbor selection instead of Version 1's
# manual sort algorithms. Does not call CustomKNN or its helpers.
# ============================================================

class GenAIKNN:
    """Version 3 - KNN algorithm developed using GenAI (Claude)."""

    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # GenAI Tool Used: Claude
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        return self

    def predict(self, X):
        # GenAI Tool Used: Claude
        if self.X_train is None:
            raise ValueError("Model must be fitted before prediction.")
        X = np.asarray(X, dtype=float)
        predictions = np.empty(X.shape[0], dtype=self.y_train.dtype)

        for i, test_point in enumerate(X):
            squared_distances = np.sum((self.X_train - test_point) ** 2, axis=1)

            if self.k < len(squared_distances):
                nearest_idx = np.argpartition(squared_distances, self.k - 1)[:self.k]
            else:
                nearest_idx = np.arange(len(squared_distances))

            # Order selected neighbors by distance, tie-broken by original index
            nearest_idx = nearest_idx[np.lexsort((nearest_idx, squared_distances[nearest_idx]))]

            neighbor_labels = self.y_train[nearest_idx]
            labels, counts = np.unique(neighbor_labels, return_counts=True)
            max_count = counts.max()
            tied_labels = set(labels[counts == max_count])

            chosen_label = None
            for idx in nearest_idx:
                if self.y_train[idx] in tied_labels:
                    chosen_label = self.y_train[idx]
                    break

            predictions[i] = chosen_label

        return predictions

    def score(self, X, y):
        # GenAI Tool Used: Claude
        predictions = self.predict(X)
        y = np.asarray(y)
        return np.sum(predictions == y) / len(y)


# ============================================================
# A3 - PERFORMANCE COMPARISON HELPERS
# ============================================================

def compute_metrics(y_true, y_pred):
    """
    Computes accuracy, precision, recall, F1 using sklearn.metrics.
    Digits dataset here is binary (classes 0 and 1); precision/recall/F1
    are calculated for the positive class (1) consistently across all versions.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    return accuracy, precision, recall, f1


def time_fit_predict(model_factory, X_train, y_train, X_test, n_runs=10):
    """
    Times fit() + predict() for a freshly constructed model, averaged over n_runs.
    Timing methodology (consistent across all 3 versions):
        - A NEW model instance is created each run via model_factory()
        - Only model.fit(X_train, y_train) followed by model.predict(X_test) is timed
        - No metric calculation, printing, or DataFrame construction happens inside the timed block
    Returns: (predictions_from_last_run, average_time_seconds)
    """
    total_time = 0.0
    predictions = None
    for _ in range(n_runs):
        model = model_factory()
        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        elapsed = time.perf_counter() - start_time
        total_time += elapsed
    average_time = total_time / n_runs
    return predictions, average_time


def evaluate_model(model_factory, X_train, y_train, X_test, y_test, n_runs=10):
    """Evaluates one model factory: returns accuracy, precision, recall, f1, avg_time."""
    predictions, average_time = time_fit_predict(model_factory, X_train, y_train, X_test, n_runs)
    accuracy, precision, recall, f1 = compute_metrics(y_test, predictions)
    return accuracy, precision, recall, f1, average_time


def run_comparison(X_train, y_train, X_test, y_test, k_values, sorting_algorithm="selection", n_runs=10):
    """
    Runs Version 1, Version 2, Version 3 across all k_values.
    Same data, same preprocessing, same split, same k values, same timing definition
    (fit + predict) for all three versions. Returns a pandas DataFrame.
    No printing or plotting inside this function.
    """
    records = []

    for k in k_values:
        v1_factory = lambda k=k: CustomKNN(k=k, algorithm=sorting_algorithm)
        acc, prec, rec, f1, avg_time = evaluate_model(v1_factory, X_train, y_train, X_test, y_test, n_runs)
        records.append({"Version": "Version 1 - Custom", "k": k, "Accuracy": acc,
                         "Precision": prec, "Recall": rec, "F1": f1, "Avg Time (s)": avg_time})

        v2_factory = lambda k=k: make_sklearn_knn(k=k)
        acc, prec, rec, f1, avg_time = evaluate_model(v2_factory, X_train, y_train, X_test, y_test, n_runs)
        records.append({"Version": "Version 2 - Sklearn", "k": k, "Accuracy": acc,
                         "Precision": prec, "Recall": rec, "F1": f1, "Avg Time (s)": avg_time})

        v3_factory = lambda k=k: GenAIKNN(k=k)
        acc, prec, rec, f1, avg_time = evaluate_model(v3_factory, X_train, y_train, X_test, y_test, n_runs)
        records.append({"Version": "Version 3 - GenAI", "k": k, "Accuracy": acc,
                         "Precision": prec, "Recall": rec, "F1": f1, "Avg Time (s)": avg_time})

    return pd.DataFrame.from_records(records)


# ============================================================
# A2 - UNIT TESTS
# GenAI Tool Used: Claude
# ============================================================

class TestEncodeData(unittest.TestCase):
    def test_normal_conversion(self):
        result = encode_data([[1, 2], [3, 4]])
        self.assertEqual(result.dtype, np.float64)

    def test_edge_empty_input(self):
        result = encode_data([[]])
        self.assertEqual(result.shape, (1, 0))


class TestCalculateMean(unittest.TestCase):
    def test_normal_case(self):
        self.assertEqual(calculate_mean([1, 2, 3]), 2.0)

    def test_edge_all_nan(self):
        self.assertEqual(calculate_mean([np.nan, np.nan]), 0)

    def test_ignores_nan_values(self):
        self.assertEqual(calculate_mean([2, np.nan, 4]), 3.0)


class TestCalculateMedian(unittest.TestCase):
    def test_odd_count(self):
        self.assertEqual(calculate_median([3, 1, 2]), 2)

    def test_even_count(self):
        self.assertEqual(calculate_median([1, 2, 3, 4]), 2.5)

    def test_edge_all_nan(self):
        self.assertEqual(calculate_median([np.nan, np.nan]), 0)


class TestCalculateMode(unittest.TestCase):
    def test_normal_case(self):
        self.assertEqual(calculate_mode([1, 2, 2, 3]), 2)

    def test_tie_returns_minimum(self):
        # 1 and 2 both appear twice; smallest should be returned
        self.assertEqual(calculate_mode([1, 1, 2, 2]), 1)

    def test_edge_all_nan(self):
        self.assertEqual(calculate_mode([np.nan, np.nan]), 0)


class TestImputeMissingValues(unittest.TestCase):
    def test_mean_imputation_normal(self):
        data = np.array([[1.0, np.nan], [3.0, 4.0]])
        result = impute_missing_values(data, method="mean")
        self.assertFalse(np.isnan(result).any())
        self.assertEqual(result[0, 1], 4.0)

    def test_median_imputation_normal(self):
        data = np.array([[np.nan], [2.0], [4.0]])
        result = impute_missing_values(data, method="median")
        self.assertEqual(result[0, 0], 3.0)

    def test_mode_imputation_normal(self):
        data = np.array([[np.nan], [2.0], [2.0], [5.0]])
        result = impute_missing_values(data, method="mode")
        self.assertEqual(result[0, 0], 2.0)

    def test_invalid_method_raises(self):
        data = np.array([[1.0, np.nan]])
        with self.assertRaises(ValueError):
            impute_missing_values(data, method="invalid")

    def test_no_missing_values_unchanged(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = impute_missing_values(data, method="mean")
        np.testing.assert_array_equal(result, data)


class TestEuclideanDistance(unittest.TestCase):
    def test_normal_case(self):
        self.assertAlmostEqual(euclidean_distance([0, 0], [3, 4]), 5.0)

    def test_edge_identical_points(self):
        self.assertEqual(euclidean_distance([1, 1], [1, 1]), 0.0)

    def test_negative_coordinates(self):
        self.assertAlmostEqual(euclidean_distance([-1, -1], [2, 3]), 5.0)


class TestSortingAlgorithms(unittest.TestCase):
    def setUp(self):
        self.items = [(5.2, 0, 0), (2.1, 1, 1), (3.7, 2, 0), (2.1, 3, 1)]

    def test_bubble_sort_normal(self):
        result = bubble_sort(self.items)
        distances = [item[0] for item in result]
        self.assertEqual(distances, sorted(distances))

    def test_selection_sort_normal(self):
        result = selection_sort(self.items)
        distances = [item[0] for item in result]
        self.assertEqual(distances, sorted(distances))

    def test_insertion_sort_normal(self):
        result = insertion_sort(self.items)
        distances = [item[0] for item in result]
        self.assertEqual(distances, sorted(distances))

    def test_edge_empty_list(self):
        self.assertEqual(bubble_sort([]), [])
        self.assertEqual(selection_sort([]), [])
        self.assertEqual(insertion_sort([]), [])

    def test_edge_single_element(self):
        single = [(1.0, 0, 0)]
        self.assertEqual(bubble_sort(single), single)
        self.assertEqual(selection_sort(single), single)
        self.assertEqual(insertion_sort(single), single)

    def test_sort_distances_dispatch(self):
        self.assertEqual(sort_distances(self.items, "bubble"), bubble_sort(self.items))
        self.assertEqual(sort_distances(self.items, "selection"), selection_sort(self.items))
        self.assertEqual(sort_distances(self.items, "insertion"), insertion_sort(self.items))

    def test_sort_distances_invalid_algorithm_raises(self):
        with self.assertRaises(ValueError):
            sort_distances(self.items, algorithm="quicksort")


class TestIdentifyNeighbors(unittest.TestCase):
    def test_normal_case_returns_k_items(self):
        distances = np.array([3.0, 1.0, 2.0])
        labels = np.array([0, 1, 0])
        neighbors = identify_neighbors(distances, labels, k=2)
        self.assertEqual(len(neighbors), 2)
        self.assertEqual(neighbors[0][2], 1)  # closest point has label 1

    def test_edge_k_larger_than_available(self):
        distances = np.array([1.0, 2.0])
        labels = np.array([0, 1])
        neighbors = identify_neighbors(distances, labels, k=5)
        self.assertEqual(len(neighbors), 2)


class TestMajorityVote(unittest.TestCase):
    def test_normal_case(self):
        neighbors = [(1.0, 0, 1), (2.0, 1, 1), (3.0, 2, 0)]
        self.assertEqual(majority_vote(neighbors), 1)

    def test_tie_broken_by_nearest_neighbor(self):
        neighbors = [(1.0, 0, 0), (2.0, 1, 1)]
        self.assertEqual(majority_vote(neighbors), 0)

    def test_single_neighbor(self):
        neighbors = [(1.0, 0, 1)]
        self.assertEqual(majority_vote(neighbors), 1)


class TestWeightedVote(unittest.TestCase):
    def test_normal_case_closer_dominates(self):
        neighbors = [(1.0, 0, 1), (10.0, 1, 0)]
        self.assertEqual(weighted_vote(neighbors), 1)

    def test_equal_distance_tie_broken_by_order(self):
        neighbors = [(2.0, 0, 0), (2.0, 1, 1)]
        self.assertEqual(weighted_vote(neighbors), 0)


class TestCustomKnnPredict(unittest.TestCase):
    def test_normal_case(self):
        X_train = np.array([[0, 0], [1, 1], [5, 5], [6, 6]])
        y_train = np.array([0, 0, 1, 1])
        predictions = custom_knn_predict(X_train, y_train, np.array([[0.1, 0.1]]), k=1)
        self.assertEqual(predictions[0], 0)


class TestWeightedKnnPredict(unittest.TestCase):
    def test_normal_case(self):
        X_train = np.array([[0, 0], [1, 1], [5, 5], [6, 6]])
        y_train = np.array([0, 0, 1, 1])
        predictions = weighted_knn_predict(X_train, y_train, np.array([[5.1, 5.1]]), k=3)
        self.assertEqual(predictions[0], 1)


class TestCustomKNNClass(unittest.TestCase):
    def setUp(self):
        self.X_train = np.array([[0, 0], [1, 1], [5, 5], [6, 6]])
        self.y_train = np.array([0, 0, 1, 1])
        self.model = CustomKNN(k=1, algorithm="selection")

    def test_predict_before_fit_raises(self):
        with self.assertRaises(ValueError):
            self.model.predict(np.array([[0, 0]]))

    def test_fit_predict_normal(self):
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(np.array([[0.1, 0.1], [5.1, 5.1]]))
        np.testing.assert_array_equal(predictions, [0, 1])

    def test_score_normal(self):
        self.model.fit(self.X_train, self.y_train)
        self.assertEqual(self.model.score(self.X_train, self.y_train), 1.0)


class TestWeightedKNNClass(unittest.TestCase):
    def setUp(self):
        self.X_train = np.array([[0, 0], [1, 1], [5, 5], [6, 6]])
        self.y_train = np.array([0, 0, 1, 1])
        self.model = WeightedKNN(k=3, algorithm="selection")

    def test_predict_before_fit_raises(self):
        with self.assertRaises(ValueError):
            self.model.predict(np.array([[0, 0]]))

    def test_fit_predict_normal(self):
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(np.array([[0.1, 0.1]]))
        self.assertEqual(predictions[0], 0)

    def test_score_normal(self):
        self.model.fit(self.X_train, self.y_train)
        self.assertEqual(self.model.score(self.X_train, self.y_train), 1.0)


class TestGenAIKNNClass(unittest.TestCase):
    def setUp(self):
        self.X_train = np.array([[0, 0], [1, 1], [5, 5], [6, 6]])
        self.y_train = np.array([0, 0, 1, 1])
        self.model = GenAIKNN(k=1)

    def test_predict_before_fit_raises(self):
        with self.assertRaises(ValueError):
            self.model.predict(np.array([[0, 0]]))

    def test_fit_predict_normal(self):
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(np.array([[0.1, 0.1], [5.1, 5.1]]))
        np.testing.assert_array_equal(predictions, [0, 1])

    def test_score_normal(self):
        self.model.fit(self.X_train, self.y_train)
        self.assertEqual(self.model.score(self.X_train, self.y_train), 1.0)

    def test_edge_k_larger_than_training_set(self):
        self.model.k = 10
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(np.array([[0, 0]]))
        self.assertEqual(len(predictions), 1)


# ============================================================
# PLOTTING
# ============================================================

def plot_accuracy_comparison(results_df, k_values, title="Accuracy Comparison"):
    plt.figure(figsize=(8, 5))
    for version_name in results_df["Version"].unique():
        subset = results_df[results_df["Version"] == version_name].sort_values("k")
        plt.plot(subset["k"], subset["Accuracy"], marker="o", label=version_name)
    plt.xlabel("Value of k")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.xticks(k_values)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_time_comparison(results_df, k_values, title="Computational Time Comparison (fit + predict)"):
    plt.figure(figsize=(8, 5))
    for version_name in results_df["Version"].unique():
        subset = results_df[results_df["Version"] == version_name].sort_values("k")
        plt.plot(subset["k"], subset["Avg Time (s)"], marker="o", label=version_name)
    plt.xlabel("Value of k")
    plt.ylabel("Average Time over 10 runs (seconds)")
    plt.title(title)
    plt.xticks(k_values)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":
    # ---- DATA PREPARATION ----
    print("DIGITS DATASET (Classes 0 and 1) - PROJECT DATA")
    selected_classes = [0, 1]
    X_raw, y = load_project_data(selected_classes)
    print("\nOriginal dataset shape (after class selection):", X_raw.shape)
    print("Classes selected:", selected_classes)
    for class_label in selected_classes:
        print("Digit", class_label, ":", np.sum(y == class_label), "samples")

    # ---- A1(a): ENCODING ----
    X = encode_data(X_raw)
    print("\nA1(a): Encoding completed.")
    print("Data type:", X.dtype)

    # ---- A1(b): DATA IMPUTATION ----
    X = impute_missing_values(X, method="mean")
    print("A1(b): Mean-based imputation completed.")
    print("Missing values:", np.isnan(X).sum())

    # ---- A1(c): DISTANCE ----
    sample_distance = euclidean_distance(X[0], X[1])
    print("\nA1(c): Euclidean Distance")
    print("Distance between first two samples:", round(sample_distance, 4))

    # ---- A1(d): SORTING ALGORITHMS ----
    # GenAI Tool Used: Claude (repetition of Lab 05 sorting demo for Lab 06 A1)
    sample_items = [(5.2, 0, 0), (2.1, 1, 1), (3.7, 2, 0), (2.1, 3, 1)]
    print("\nA1(d): Sorting Algorithms")
    print("Bubble Sort:", bubble_sort(sample_items))
    print("Selection Sort:", selection_sort(sample_items))
    print("Insertion Sort:", insertion_sort(sample_items))

    # ---- TRAIN-TEST SPLIT (same for all versions) ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    print("\nTRAIN-TEST SPLIT")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # ---- A1(e) AND A1(f): NEIGHBOR IDENTIFICATION + MAJORITY VOTE DEMO ----
    # GenAI Tool Used: Claude (repetition of Lab 05 demo for Lab 06 A1)
    print("\nA1(e)/(f): CUSTOM KNN MODULES DEMO")
    demonstration_distances = np.sqrt(np.sum((X_train - X_test[0]) ** 2, axis=1))
    demonstration_neighbors = identify_neighbors(demonstration_distances, y_train, k=3, algorithm="selection")
    demonstration_prediction = majority_vote(demonstration_neighbors)
    print("First test sample true class:", y_test[0])
    print("3 nearest neighbors' labels:", [n[2] for n in demonstration_neighbors])
    print("Neighbor distances:", [round(n[0], 4) for n in demonstration_neighbors])
    print("Predicted class:", demonstration_prediction)

    # ---- A4/A5/A6 (Lab05): SCIKIT-LEARN KNN - FIT, ACCURACY, PREDICT ----
    # GenAI Tool Used: Claude (repetition of Lab 05 sklearn KNN demo for Lab 06 A1)
    print("\nSCIKIT-LEARN KNN (k=3)")
    sklearn_knn = make_sklearn_knn(k=3)
    start_time = time.perf_counter()
    sklearn_knn.fit(X_train, y_train)
    sklearn_train_time = time.perf_counter() - start_time
    print("Training time:", round(sklearn_train_time, 6), "seconds")
    sklearn_accuracy = sklearn_knn.score(X_test, y_test)
    print("Scikit-learn KNN Accuracy:", round(sklearn_accuracy, 4))
    sklearn_predictions = sklearn_knn.predict(X_test)
    print("First 20 actual labels:   ", y_test[:20])
    print("First 20 predicted labels:", sklearn_predictions[:20])

    # ---- A7 (Lab05): CUSTOM KNN (VERSION 1) ----
    # GenAI Tool Used: Claude (repetition of Lab 05 custom KNN demo for Lab 06 A1)
    print("\nCUSTOM KNN - VERSION 1 (k=3)")
    custom_model = CustomKNN(k=3, algorithm="selection")
    start_time = time.perf_counter()
    custom_model.fit(X_train, y_train)
    custom_fit_time = time.perf_counter() - start_time
    start_time = time.perf_counter()
    custom_predictions = custom_model.predict(X_test)
    custom_predict_time = time.perf_counter() - start_time
    custom_accuracy = custom_model.score(X_test, y_test)
    print("Custom KNN Accuracy:", round(custom_accuracy, 4))
    print("Custom KNN prediction time:", round(custom_predict_time, 6), "seconds")
    print("First 20 custom predictions:", custom_predictions[:20])

    # ---- A2 (Lab05): WEIGHTED KNN ----
    # GenAI Tool Used: Claude (repetition of Lab 05 weighted KNN demo for Lab 06 A1)
    print("\nWEIGHTED KNN (k=3)")
    weighted_model = WeightedKNN(k=3, algorithm="selection")
    start_time = time.perf_counter()
    weighted_model.fit(X_train, y_train)
    weighted_predictions = weighted_model.predict(X_test)
    weighted_time = time.perf_counter() - start_time
    weighted_accuracy = weighted_model.score(X_test, y_test)
    print("Weighted KNN Accuracy:", round(weighted_accuracy, 4))
    print("Weighted KNN prediction time:", round(weighted_time, 6), "seconds")
    print("First 20 weighted KNN predictions:", weighted_predictions[:20])

    # ---- A8 (Lab05): CUSTOM vs SKLEARN ACROSS K VALUES ----
    # GenAI Tool Used: Claude (repetition of Lab 05 k-sweep comparison for Lab 06 A1)
    print("\nCUSTOM KNN vs SCIKIT-LEARN KNN ACROSS K VALUES")
    k_values = [1, 3, 5, 7, 9]
    sorting_algorithm = "selection"
    custom_accuracies, sklearn_accuracies = [], []
    custom_times, sklearn_times = [], []
    print("{:<8}{:<20}{:<20}".format("K", "Custom KNN", "Scikit-learn"))
    for k_value in k_values:
        custom_model_k = CustomKNN(k=k_value, algorithm=sorting_algorithm)
        custom_model_k.fit(X_train, y_train)
        start_time = time.perf_counter()
        custom_accuracy_k = custom_model_k.score(X_test, y_test)
        custom_time_k = time.perf_counter() - start_time

        sklearn_model_k = make_sklearn_knn(k=k_value)
        start_time = time.perf_counter()
        sklearn_model_k.fit(X_train, y_train)
        sklearn_accuracy_k = sklearn_model_k.score(X_test, y_test)
        sklearn_time_k = time.perf_counter() - start_time

        custom_accuracies.append(custom_accuracy_k)
        sklearn_accuracies.append(sklearn_accuracy_k)
        custom_times.append(custom_time_k)
        sklearn_times.append(sklearn_time_k)
        print("{:<8}{:<20}{:<20}".format(k_value, round(custom_accuracy_k, 4), round(sklearn_accuracy_k, 4)))

    best_custom_index = int(np.argmax(custom_accuracies))
    best_sklearn_index = int(np.argmax(sklearn_accuracies))
    print("\nBest K for Custom KNN:", k_values[best_custom_index],
          "| Accuracy:", round(custom_accuracies[best_custom_index] * 100, 2), "%")
    print("Best K for Scikit-learn KNN:", k_values[best_sklearn_index],
          "| Accuracy:", round(sklearn_accuracies[best_sklearn_index] * 100, 2), "%")

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, custom_accuracies, marker="o", label="Custom KNN (V1)")
    plt.plot(k_values, sklearn_accuracies, marker="s", label="Scikit-learn KNN (V2)")
    plt.xlabel("Value of k")
    plt.ylabel("Accuracy")
    plt.title("A1 Repetition: Custom KNN vs Scikit-learn KNN")
    plt.xticks(k_values)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, custom_times, marker="o", label="Custom KNN (V1)")
    plt.plot(k_values, sklearn_times, marker="s", label="Scikit-learn KNN (V2)")
    plt.xlabel("Value of k")
    plt.ylabel("Execution Time (seconds)")
    plt.title("A1 Repetition: Execution Time Comparison")
    plt.xticks(k_values)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- PREDICTION COMPARISON FOR k = 3 (Lab05 style) ----
    print("\nPREDICTION COMPARISON FOR k = 3 (Custom vs Scikit-learn)")
    prediction_comparison = (custom_predictions == sklearn_predictions)
    print("Number of identical predictions:", np.sum(prediction_comparison))
    print("Total test samples:", len(y_test))
    print("Predictions identical:", np.all(prediction_comparison))

    # ============================================================
    # A3: THREE-WAY PERFORMANCE COMPARISON (VERSION 1 vs 2 vs 3)
    # ============================================================
    print("\nA3: THREE-WAY PERFORMANCE COMPARISON (10 runs averaged, fit+predict timed)")
    results_df = run_comparison(X_train, y_train, X_test, y_test, k_values,
                                 sorting_algorithm=sorting_algorithm, n_runs=10)
    pd.set_option("display.float_format", lambda x: f"{x:.6f}")
    print("\n", results_df.to_string(index=False))

    plot_accuracy_comparison(results_df, k_values,
                              title="A3: Accuracy Comparison - Version 1 vs 2 vs 3")
    plot_time_comparison(results_df, k_values,
                          title="A3: Computational Time Comparison - Version 1 vs 2 vs 3")

    # ============================================================
    # A2: RUN UNIT TESTS
    # ============================================================
    print("\nA2: RUNNING UNIT TESTS")
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    test_classes = [
        TestEncodeData, TestCalculateMean, TestCalculateMedian, TestCalculateMode,
        TestImputeMissingValues, TestEuclideanDistance, TestSortingAlgorithms,
        TestIdentifyNeighbors, TestMajorityVote, TestWeightedVote,
        TestCustomKnnPredict, TestWeightedKnnPredict,
        TestCustomKNNClass, TestWeightedKNNClass, TestGenAIKNNClass
    ]
    for test_class in test_classes:
        test_suite.addTests(test_loader.loadTestsFromTestCase(test_class))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite)