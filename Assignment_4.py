# ============================================================
# LAB 4 — MACHINE LEARNING
# Combined Modular Implementation + Unit Tests + K-Means Comparison
# GenAI Tool Used: Claude
# ============================================================

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski as scipy_minkowski
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# ============================================================
# SECTION 1: A1 — Label Encoding, One-Hot Encoding, Dimensionality
# ============================================================

# GenAI Tool Used: Claude
def load_data(filepath, sheet_name="marketing_campaign"):
    """Loads the marketing_campaign worksheet from the given Excel file."""
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    return df


# GenAI Tool Used: Claude
def label_encoding(df, categorical_columns):
    """
    Applies Label Encoding to specified categorical columns.
    Returns dict: {column: (encoded_values, fitted_encoder)}
    """
    encoded_data = {}
    for col in categorical_columns:
        le = LabelEncoder()
        encoded_values = le.fit_transform(df[col].astype(str))
        encoded_data[col] = (encoded_values, le)
    return encoded_data


# GenAI Tool Used: Claude
def one_hot_encoding(df, categorical_columns):
    """
    Applies One-Hot Encoding to specified categorical columns.
    Returns (encoded_array, feature_names, fitted_encoder).
    """
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_array = ohe.fit_transform(df[categorical_columns].astype(str))
    feature_names = ohe.get_feature_names_out(categorical_columns)
    return encoded_array, feature_names, ohe


# GenAI Tool Used: Claude
def recreate_dataset_label(df, encoded_data, categorical_columns):
    """Replaces original categorical columns with label-encoded values."""
    df_label = df.copy()
    for col in categorical_columns:
        df_label[col] = encoded_data[col][0]
    return df_label


# GenAI Tool Used: Claude
def recreate_dataset_one_hot(df, encoded_array, feature_names, categorical_columns):
    """Drops original categorical columns and appends one-hot columns."""
    df_remaining = df.drop(columns=categorical_columns).reset_index(drop=True)
    df_encoded = pd.DataFrame(encoded_array, columns=feature_names)
    df_one_hot = pd.concat([df_remaining, df_encoded], axis=1)
    return df_one_hot


# GenAI Tool Used: Claude
def feature_dimension(df_original, df_label, df_one_hot):
    """Compares number of columns across dataset variants."""
    dimensions = {
        "original": df_original.shape[1],
        "label_encoded": df_label.shape[1],
        "one_hot_encoded": df_one_hot.shape[1]
    }
    return dimensions


# ============================================================
# SECTION 2: A2/A3 — Minkowski Distance (custom, p-sweep, scipy compare)
# ============================================================

# GenAI Tool Used: Claude
def minkowski_distance(vector1, vector2, p):
    """
    Generalized Minkowski distance between two 1D vectors for order p.
    Formula: (sum(|x_i - y_i|^p))^(1/p)
    """
    vector1 = np.asarray(vector1, dtype=float)
    vector2 = np.asarray(vector2, dtype=float)
    diff = np.abs(vector1 - vector2)
    distance = np.sum(diff ** p) ** (1.0 / p)
    return distance


# GenAI Tool Used: Claude
def minkowski_plot(vector1, vector2, p_values):
    """
    Computes Minkowski distances across a range of p values.
    Returns (p_list, distances) only — plotting happens in main().
    """
    distances = [minkowski_distance(vector1, vector2, p) for p in p_values]
    return list(p_values), distances


# GenAI Tool Used: Claude
def scipy_minkowski_distance(vector1, vector2, p):
    """Computes Minkowski distance using scipy, for comparison."""
    return scipy_minkowski(vector1, vector2, p)


# ============================================================
# SECTION 3: A4 — Custom dot product and Euclidean norm (vectorized)
# ============================================================

# GenAI Tool Used: Claude
def dot_product(vector1, vector2):
    """Dot product via vectorized elementwise multiply + sum."""
    vector1 = np.asarray(vector1, dtype=float)
    vector2 = np.asarray(vector2, dtype=float)
    return np.sum(vector1 * vector2)


# GenAI Tool Used: Claude
def euclidean_norm(vector):
    """Euclidean (L2) norm via vectorized operations."""
    vector = np.asarray(vector, dtype=float)
    return np.sqrt(np.sum(vector ** 2))


# ============================================================
# SECTION 4: A5/A6 — Custom mean, variance, std + comparison
# ============================================================

# GenAI Tool Used: Claude
def mean(data):
    """Mean via vectorized sum."""
    data = np.asarray(data, dtype=float)
    return np.sum(data) / len(data)


# GenAI Tool Used: Claude
def variance(data):
    """Population variance (divide by N) via vectorized operations."""
    data = np.asarray(data, dtype=float)
    m = mean(data)
    return np.sum((data - m) ** 2) / len(data)


# GenAI Tool Used: Claude
def standard_deviation(data):
    """Standard deviation (sqrt of population variance)."""
    return variance(data) ** 0.5


# GenAI Tool Used: Claude
def dataset_statistics(df, numeric_columns):
    """Applies custom mean/variance/std_dev to each numeric column."""
    stats = {}
    for col in numeric_columns:
        col_data = df[col].dropna().values
        stats[col] = {
            "mean": mean(col_data),
            "variance": variance(col_data),
            "std_dev": standard_deviation(col_data)
        }
    return stats


# GenAI Tool Used: Claude
def numpy_statistics(df, numeric_columns):
    """Computes mean/variance/std_dev using NumPy, for comparison."""
    stats = {}
    for col in numeric_columns:
        col_data = df[col].dropna().values
        stats[col] = {
            "mean": np.mean(col_data),
            "variance": np.var(col_data),
            "std_dev": np.std(col_data)
        }
    return stats


# ============================================================
# SECTION 5: A7 — Histogram analysis
# ============================================================

# GenAI Tool Used: Claude
def histogram(data, bins=10):
    """Computes histogram bin counts and bin edges (no plotting inside)."""
    data = np.asarray(data, dtype=float)
    counts, bin_edges = np.histogram(data, bins=bins)
    return counts, bin_edges


# ============================================================
# SECTION 6: A8 — K-Means VERSION 1 (your original custom style)
# ============================================================

# GenAI Tool Used: Claude
def assign_clusters(data, centroids):
    """Assigns each point to the nearest centroid using Euclidean distance."""
    n_samples = data.shape[0]
    k = centroids.shape[0]
    distances = np.zeros((n_samples, k))

    for i in range(k):
        diff = data - centroids[i]
        distances[:, i] = np.sqrt(np.sum(diff ** 2, axis=1))

    cluster_labels = np.argmin(distances, axis=1)
    return cluster_labels


# GenAI Tool Used: Claude
def calculate_centroids(data, cluster_labels, k):
    """Recomputes centroids as the mean of points in each cluster."""
    n_features = data.shape[1]
    centroids = np.zeros((k, n_features))

    for i in range(k):
        cluster_points = data[cluster_labels == i]
        if len(cluster_points) > 0:
            centroids[i] = np.mean(cluster_points, axis=0)
        # if a cluster loses all its points, its centroid stays at zero
        # (simple fallback strategy — Version 2 below handles this differently)

    return centroids


# GenAI Tool Used: Claude
def k_means(data, k, max_iterations=100, tol=1e-4, random_seed=42):
    """
    K-Means VERSION 1: random initialization, per-centroid distance loop,
    aggregate (summed) shift convergence check.
    Returns (final_cluster_labels, final_centroids, n_iterations_run)
    """
    rng = np.random.default_rng(random_seed)
    n_samples = data.shape[0]

    initial_indices = rng.choice(n_samples, size=k, replace=False)
    centroids = data[initial_indices].copy()

    for iteration in range(max_iterations):
        cluster_labels = assign_clusters(data, centroids)
        new_centroids = calculate_centroids(data, cluster_labels, k)

        shift = np.sqrt(np.sum((new_centroids - centroids) ** 2))
        centroids = new_centroids

        if shift < tol:
            break

    n_iterations_run = iteration + 1
    return cluster_labels, centroids, n_iterations_run


# ============================================================
# SECTION 7: A3 — K-Means VERSION 2 (AI-generated, k-means++ style)
# Kept fully separate from Version 1 for side-by-side comparison
# ============================================================

# GenAI Tool Used: Claude
def initialize_centroids_v2(data, k, random_seed=42):
    """
    k-means++ initialization: first centroid random, subsequent centroids
    chosen with probability proportional to squared distance from the
    nearest already-chosen centroid.
    """
    rng = np.random.default_rng(random_seed)
    n_samples = data.shape[0]
    centroids = np.zeros((k, data.shape[1]))

    first_idx = rng.integers(0, n_samples)
    centroids[0] = data[first_idx]

    for i in range(1, k):
        distances_to_nearest = np.min(
            np.array([np.sum((data - c) ** 2, axis=1) for c in centroids[:i]]),
            axis=0
        )
        probabilities = distances_to_nearest / np.sum(distances_to_nearest)
        next_idx = rng.choice(n_samples, p=probabilities)
        centroids[i] = data[next_idx]

    return centroids


# GenAI Tool Used: Claude
def compute_euclidean_distances_v2(data, centroids):
    """Fully vectorized Euclidean distance matrix (n_samples x k) via broadcasting."""
    diff = data[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    return distances


# GenAI Tool Used: Claude
def assign_clusters_v2(data, centroids):
    """Assigns each point to its nearest centroid (Version 2)."""
    distances = compute_euclidean_distances_v2(data, centroids)
    return np.argmin(distances, axis=1)


# GenAI Tool Used: Claude
def update_centroids_v2(data, cluster_labels, k, previous_centroids):
    """
    Recomputes centroids as the mean of assigned points.
    Empty clusters fall back to their previous centroid position
    instead of collapsing to zero.
    """
    n_features = data.shape[1]
    new_centroids = np.zeros((k, n_features))

    for i in range(k):
        assigned_points = data[cluster_labels == i]
        if len(assigned_points) > 0:
            new_centroids[i] = np.mean(assigned_points, axis=0)
        else:
            new_centroids[i] = previous_centroids[i]

    return new_centroids


# GenAI Tool Used: Claude
def has_converged_v2(old_centroids, new_centroids, tol=1e-4):
    """Convergence based on the MAXIMUM single-centroid shift (stricter than sum)."""
    shifts = np.sqrt(np.sum((new_centroids - old_centroids) ** 2, axis=1))
    return np.max(shifts) < tol


# GenAI Tool Used: Claude
def k_means_v2(data, k, max_iterations=100, tol=1e-4, random_seed=42):
    """
    K-Means VERSION 2: k-means++ initialization, vectorized distance matrix,
    empty-cluster fallback, max-shift convergence check.
    Same input/output contract as k_means() for direct comparison.
    """
    centroids = initialize_centroids_v2(data, k, random_seed=random_seed)

    for iteration in range(max_iterations):
        cluster_labels = assign_clusters_v2(data, centroids)
        new_centroids = update_centroids_v2(data, cluster_labels, k, centroids)

        if has_converged_v2(centroids, new_centroids, tol=tol):
            centroids = new_centroids
            break

        centroids = new_centroids

    n_iterations_run = iteration + 1
    return cluster_labels, centroids, n_iterations_run


# GenAI Tool Used: Claude
def compare_kmeans_versions(data, k, random_seed=42):
    """
    Runs both K-Means versions on identical data and returns both results
    together. Contains no prints — comparison/reporting happens in main().
    """
    labels_v1, centroids_v1, iters_v1 = k_means(data, k, random_seed=random_seed)
    labels_v2, centroids_v2, iters_v2 = k_means_v2(data, k, random_seed=random_seed)

    return {
        "version1": (labels_v1, centroids_v1, iters_v1),
        "version2": (labels_v2, centroids_v2, iters_v2)
    }


# ============================================================
# SECTION 8: UNIT TESTS FOR ALL MODULAR FUNCTIONS
# ============================================================

# GenAI Tool Used: Claude
class TestEncodingFunctions(unittest.TestCase):
    """Tests for load_data, label_encoding, one_hot_encoding, recreate_*, feature_dimension."""

    def setUp(self):
        self.df = pd.DataFrame({
            "Education": ["Graduation", "PhD", "Graduation", "Master"],
            "Marital_Status": ["Married", "Single", "Married", "Divorced"],
            "Income": [50000, 60000, 55000, 45000]
        })
        self.categorical_columns = ["Education", "Marital_Status"]

    @patch("pandas.read_excel")
    def test_load_data_calls_read_excel_with_correct_args(self, mock_read_excel):
        mock_read_excel.return_value = self.df
        result = load_data("Lab Session Data.xlsx", sheet_name="marketing_campaign")
        mock_read_excel.assert_called_once_with(
            "Lab Session Data.xlsx", sheet_name="marketing_campaign"
        )
        pd.testing.assert_frame_equal(result, self.df)

    @patch("pandas.read_excel")
    def test_load_data_uses_default_sheet_name(self, mock_read_excel):
        mock_read_excel.return_value = pd.DataFrame({"A": [1]})
        load_data("some_file.xlsx")
        _, kwargs = mock_read_excel.call_args
        self.assertEqual(kwargs.get("sheet_name"), "marketing_campaign")

    def test_label_encoding_output_shape(self):
        encoded = label_encoding(self.df, self.categorical_columns)
        self.assertEqual(len(encoded["Education"][0]), len(self.df))

    def test_label_encoding_consistent_classes(self):
        encoded = label_encoding(self.df, self.categorical_columns)
        values = encoded["Education"][0]
        self.assertEqual(values[0], values[2])  # both "Graduation"

    def test_one_hot_encoding_column_count(self):
        encoded_array, feature_names, _ = one_hot_encoding(self.df, self.categorical_columns)
        self.assertEqual(encoded_array.shape[1], 6)  # 3 Education + 3 Marital_Status

    def test_one_hot_encoding_row_sums(self):
        encoded_array, _, _ = one_hot_encoding(self.df, self.categorical_columns)
        row_sums = encoded_array.sum(axis=1)
        self.assertTrue(np.all(row_sums == 2))

    def test_recreate_dataset_label_replaces_with_integers(self):
        encoded_data = label_encoding(self.df, self.categorical_columns)
        df_label = recreate_dataset_label(self.df, encoded_data, self.categorical_columns)
        self.assertTrue(np.issubdtype(df_label["Education"].dtype, np.integer))

    def test_recreate_dataset_label_preserves_other_columns(self):
        encoded_data = label_encoding(self.df, self.categorical_columns)
        df_label = recreate_dataset_label(self.df, encoded_data, self.categorical_columns)
        pd.testing.assert_series_equal(df_label["Income"], self.df["Income"])

    def test_recreate_dataset_one_hot_drops_categorical_columns(self):
        encoded_array, feature_names, _ = one_hot_encoding(self.df, self.categorical_columns)
        df_one_hot = recreate_dataset_one_hot(
            self.df, encoded_array, feature_names, self.categorical_columns
        )
        self.assertNotIn("Education", df_one_hot.columns)
        self.assertNotIn("Marital_Status", df_one_hot.columns)

    def test_recreate_dataset_one_hot_row_count(self):
        encoded_array, feature_names, _ = one_hot_encoding(self.df, self.categorical_columns)
        df_one_hot = recreate_dataset_one_hot(
            self.df, encoded_array, feature_names, self.categorical_columns
        )
        self.assertEqual(len(df_one_hot), len(self.df))

    def test_feature_dimension_reports_correct_counts(self):
        encoded_data = label_encoding(self.df, self.categorical_columns)
        df_label = recreate_dataset_label(self.df, encoded_data, self.categorical_columns)
        encoded_array, feature_names, _ = one_hot_encoding(self.df, self.categorical_columns)
        df_one_hot = recreate_dataset_one_hot(
            self.df, encoded_array, feature_names, self.categorical_columns
        )
        dims = feature_dimension(self.df, df_label, df_one_hot)
        self.assertEqual(dims["original"], 3)
        self.assertEqual(dims["label_encoded"], 3)
        self.assertEqual(dims["one_hot_encoded"], 1 + 6)


# GenAI Tool Used: Claude
class TestMinkowskiDistance(unittest.TestCase):
    """Tests for minkowski_distance, minkowski_plot, scipy_minkowski_distance."""

    def setUp(self):
        self.v1 = np.array([1.0, 2.0, 3.0])
        self.v2 = np.array([4.0, 6.0, 8.0])

    def test_minkowski_p1_matches_manhattan(self):
        expected = np.sum(np.abs(self.v1 - self.v2))
        self.assertAlmostEqual(minkowski_distance(self.v1, self.v2, p=1), expected, places=6)

    def test_minkowski_p2_matches_euclidean(self):
        expected = np.sqrt(np.sum((self.v1 - self.v2) ** 2))
        self.assertAlmostEqual(minkowski_distance(self.v1, self.v2, p=2), expected, places=6)

    def test_minkowski_matches_scipy_across_p_values(self):
        for p in range(1, 11):
            self.assertAlmostEqual(
                minkowski_distance(self.v1, self.v2, p),
                scipy_minkowski(self.v1, self.v2, p),
                places=6
            )

    def test_minkowski_distance_zero_for_identical_vectors(self):
        self.assertAlmostEqual(minkowski_distance(self.v1, self.v1, p=3), 0.0, places=6)

    def test_minkowski_plot_returns_correct_lengths(self):
        p_list, distances = minkowski_plot(self.v1, self.v2, range(1, 11))
        self.assertEqual(len(p_list), 10)
        self.assertEqual(len(distances), 10)

    def test_scipy_wrapper_matches_custom_implementation(self):
        for p in [1, 2, 3, 7]:
            self.assertAlmostEqual(
                minkowski_distance(self.v1, self.v2, p),
                scipy_minkowski_distance(self.v1, self.v2, p),
                places=6
            )

    def test_scipy_wrapper_known_value(self):
        v1 = np.array([0.0, 0.0])
        v2 = np.array([3.0, 4.0])
        self.assertAlmostEqual(scipy_minkowski_distance(v1, v2, p=2), 5.0, places=6)


# GenAI Tool Used: Claude
class TestDotProductAndNorm(unittest.TestCase):
    """Tests for dot_product and euclidean_norm."""

    def setUp(self):
        self.v1 = np.array([1.0, 2.0, 3.0])
        self.v2 = np.array([4.0, 5.0, 6.0])

    def test_dot_product_matches_numpy(self):
        self.assertAlmostEqual(dot_product(self.v1, self.v2), np.dot(self.v1, self.v2), places=6)

    def test_dot_product_with_zero_vector(self):
        self.assertAlmostEqual(dot_product(self.v1, np.zeros(3)), 0.0, places=6)

    def test_euclidean_norm_matches_numpy(self):
        self.assertAlmostEqual(euclidean_norm(self.v1), np.linalg.norm(self.v1), places=6)

    def test_euclidean_norm_of_zero_vector(self):
        self.assertAlmostEqual(euclidean_norm(np.zeros(5)), 0.0, places=6)


# GenAI Tool Used: Claude
class TestStatisticsFunctions(unittest.TestCase):
    """Tests for mean, variance, standard_deviation, dataset_statistics, numpy_statistics."""

    def setUp(self):
        self.data = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        self.df = pd.DataFrame({
            "Income": [50000, 60000, 55000, 45000, 70000],
            "Age": [25, 30, 35, 40, 45]
        })
        self.numeric_columns = ["Income", "Age"]

    def test_mean_matches_numpy(self):
        self.assertAlmostEqual(mean(self.data), np.mean(self.data), places=6)

    def test_variance_matches_numpy(self):
        self.assertAlmostEqual(variance(self.data), np.var(self.data), places=6)

    def test_standard_deviation_matches_numpy(self):
        self.assertAlmostEqual(standard_deviation(self.data), np.std(self.data), places=6)

    def test_variance_is_square_of_std_dev(self):
        self.assertAlmostEqual(standard_deviation(self.data) ** 2, variance(self.data), places=6)

    def test_variance_of_constant_array_is_zero(self):
        self.assertAlmostEqual(variance(np.array([5.0, 5.0, 5.0])), 0.0, places=6)

    def test_dataset_statistics_matches_numpy_statistics(self):
        custom_stats = dataset_statistics(self.df, self.numeric_columns)
        numpy_stats = numpy_statistics(self.df, self.numeric_columns)
        for col in self.numeric_columns:
            self.assertAlmostEqual(custom_stats[col]["mean"], numpy_stats[col]["mean"], places=6)
            self.assertAlmostEqual(custom_stats[col]["variance"], numpy_stats[col]["variance"], places=6)

    def test_dataset_statistics_handles_missing_values(self):
        df_nan = self.df.copy()
        df_nan.loc[0, "Income"] = np.nan
        stats = dataset_statistics(df_nan, ["Income"])
        expected_mean = df_nan["Income"].dropna().mean()
        self.assertAlmostEqual(stats["Income"]["mean"], expected_mean, places=6)


# GenAI Tool Used: Claude
class TestHistogram(unittest.TestCase):
    """Tests for histogram."""

    def test_histogram_counts_sum_to_data_length(self):
        data = np.array([1, 2, 2, 3, 3, 3, 4, 5])
        counts, _ = histogram(data, bins=5)
        self.assertEqual(np.sum(counts), len(data))

    def test_histogram_bin_edges_length(self):
        _, bin_edges = histogram(np.array([1, 2, 3, 4, 5]), bins=4)
        self.assertEqual(len(bin_edges), 5)

    def test_histogram_matches_numpy_directly(self):
        data = np.random.default_rng(0).normal(size=100)
        counts, edges = histogram(data, bins=10)
        expected_counts, expected_edges = np.histogram(data, bins=10)
        np.testing.assert_array_equal(counts, expected_counts)
        np.testing.assert_array_almost_equal(edges, expected_edges)


# GenAI Tool Used: Claude
class TestKMeansVersion1(unittest.TestCase):
    """Tests for assign_clusters, calculate_centroids, k_means (Version 1)."""

    def setUp(self):
        self.data = np.array([
            [0.0, 0.0], [0.1, 0.1], [0.2, 0.0],
            [10.0, 10.0], [10.1, 9.9], [9.9, 10.1]
        ])
        self.centroids = np.array([[0.0, 0.0], [10.0, 10.0]])

    def test_assign_clusters_correct_labels(self):
        labels = assign_clusters(self.data, self.centroids)
        np.testing.assert_array_equal(labels, np.array([0, 0, 0, 1, 1, 1]))

    def test_calculate_centroids_correct_means(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        centroids = calculate_centroids(self.data, labels, k=2)
        np.testing.assert_array_almost_equal(centroids[0], self.data[:3].mean(axis=0))
        np.testing.assert_array_almost_equal(centroids[1], self.data[3:].mean(axis=0))

    def test_kmeans_v1_converges_to_two_clusters(self):
        labels, centroids, n_iters = k_means(self.data, k=2, random_seed=1)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[1], labels[2])
        self.assertEqual(labels[3], labels[4])
        self.assertNotEqual(labels[0], labels[3])

    def test_kmeans_v1_returns_k_centroids(self):
        _, centroids, _ = k_means(self.data, k=2, random_seed=1)
        self.assertEqual(centroids.shape[0], 2)


# GenAI Tool Used: Claude
class TestKMeansVersion2(unittest.TestCase):
    """Tests for initialize_centroids_v2, assign_clusters_v2, update_centroids_v2, k_means_v2."""

    def setUp(self):
        self.data = np.array([
            [0.0, 0.0], [0.1, 0.1], [0.2, 0.0],
            [10.0, 10.0], [10.1, 9.9], [9.9, 10.1]
        ])

    def test_initialize_centroids_v2_returns_correct_shape(self):
        centroids = initialize_centroids_v2(self.data, k=2, random_seed=1)
        self.assertEqual(centroids.shape, (2, 2))

    def test_assign_clusters_v2_correct_labels(self):
        centroids = np.array([[0.0, 0.0], [10.0, 10.0]])
        labels = assign_clusters_v2(self.data, centroids)
        np.testing.assert_array_equal(labels, np.array([0, 0, 0, 1, 1, 1]))

    def test_update_centroids_v2_correct_means(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        prev_centroids = np.array([[0.0, 0.0], [10.0, 10.0]])
        centroids = update_centroids_v2(self.data, labels, k=2, previous_centroids=prev_centroids)
        np.testing.assert_array_almost_equal(centroids[0], self.data[:3].mean(axis=0))

    def test_update_centroids_v2_empty_cluster_fallback(self):
        # All points assigned to cluster 0 -> cluster 1 is empty
        labels = np.array([0, 0, 0, 0, 0, 0])
        prev_centroids = np.array([[1.0, 1.0], [9.0, 9.0]])
        centroids = update_centroids_v2(self.data, labels, k=2, previous_centroids=prev_centroids)
        # Empty cluster (index 1) should fall back to its previous position
        np.testing.assert_array_almost_equal(centroids[1], prev_centroids[1])

    def test_has_converged_v2_true_for_identical_centroids(self):
        centroids = np.array([[1.0, 1.0], [2.0, 2.0]])
        self.assertTrue(has_converged_v2(centroids, centroids, tol=1e-4))

    def test_has_converged_v2_false_for_large_shift(self):
        old = np.array([[0.0, 0.0], [10.0, 10.0]])
        new = np.array([[5.0, 5.0], [10.0, 10.0]])
        self.assertFalse(has_converged_v2(old, new, tol=1e-4))

    def test_kmeans_v2_converges_to_two_clusters(self):
        labels, centroids, n_iters = k_means_v2(self.data, k=2, random_seed=1)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[3], labels[4])
        self.assertNotEqual(labels[0], labels[3])

    def test_kmeans_v2_returns_k_centroids(self):
        _, centroids, _ = k_means_v2(self.data, k=2, random_seed=1)
        self.assertEqual(centroids.shape[0], 2)

    def test_compare_kmeans_versions_returns_both_results(self):
        results = compare_kmeans_versions(self.data, k=2, random_seed=1)
        self.assertIn("version1", results)
        self.assertIn("version2", results)
        self.assertEqual(results["version1"][1].shape[0], 2)
        self.assertEqual(results["version2"][1].shape[0], 2)


# ============================================================
# SECTION 9: MAIN PROGRAM — all print statements live here
# ============================================================

# GenAI Tool Used: Claude
def main():
    filepath = "Lab Session Data.xlsx"
    categorical_columns = ["Education", "Marital_Status"]

    # ---------------- A1: Encoding & Dimensionality ----------------
    df = load_data(filepath)
    print("Dataset loaded. Shape:", df.shape)
    print(df.head())

    encoded_data = label_encoding(df, categorical_columns)
    for col, (values, encoder) in encoded_data.items():
        print(f"\nLabel Encoding for '{col}': classes={encoder.classes_}, sample={values[:5]}")

    ohe_array, ohe_feature_names, _ = one_hot_encoding(df, categorical_columns)
    print("\nOne-Hot feature names:", ohe_feature_names)

    df_label = recreate_dataset_label(df, encoded_data, categorical_columns)
    df_one_hot = recreate_dataset_one_hot(df, ohe_array, ohe_feature_names, categorical_columns)

    dims = feature_dimension(df, df_label, df_one_hot)
    print("\nFeature Dimensionality Comparison:", dims)

    # ---------------- A2: Minkowski Distance ----------------
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    vector1 = df_label.loc[0, numeric_columns].values.astype(float)
    vector2 = df_label.loc[1, numeric_columns].values.astype(float)

    p_values = range(1, 11)
    p_list, distances = minkowski_plot(vector1, vector2, p_values)
    print("\nMinkowski distances (p=1 to 10):")
    for p, d in zip(p_list, distances):
        scipy_val = scipy_minkowski_distance(vector1, vector2, p)
        print(f"  p={p}: custom={d:.4f}, scipy={scipy_val:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(p_list, distances, marker='o')
    plt.title("Minkowski Distance vs p (row0 vs row1)")
    plt.xlabel("p value")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.show()

    # ---------------- A4: Dot product & Euclidean norm ----------------
    print(f"\nDot product: custom={dot_product(vector1, vector2):.4f}, "
          f"numpy={np.dot(vector1, vector2):.4f}")
    print(f"Euclidean norm: custom={euclidean_norm(vector1):.4f}, "
          f"numpy={np.linalg.norm(vector1):.4f}")

    # ---------------- A5/A6: Statistics ----------------
    custom_stats = dataset_statistics(df, numeric_columns)
    numpy_stats = numpy_statistics(df, numeric_columns)
    print("\nStatistics comparison (first 3 numeric columns):")
    for col in numeric_columns[:3]:
        print(f"  {col}: custom={custom_stats[col]}, numpy={numpy_stats[col]}")

    # ---------------- A7: Histogram ----------------
    feature_for_hist = "Income" if "Income" in df.columns else numeric_columns[0]
    hist_data = df[feature_for_hist].dropna().values
    counts, bin_edges = histogram(hist_data, bins=10)
    print(f"\nHistogram for '{feature_for_hist}': counts={counts}")

    plt.figure(figsize=(8, 5))
    plt.hist(hist_data, bins=10, edgecolor='black')
    plt.title(f"Histogram of {feature_for_hist}")
    plt.xlabel(feature_for_hist)
    plt.ylabel("Frequency")
    plt.show()

    # ---------------- A3: K-Means Version 1 vs Version 2 ----------------
    cluster_data = df[numeric_columns].dropna().values
    k = 3

    results = compare_kmeans_versions(cluster_data, k=k, random_seed=42)
    labels_v1, centroids_v1, iters_v1 = results["version1"]
    labels_v2, centroids_v2, iters_v2 = results["version2"]

    print(f"\nK-Means Version 1: converged in {iters_v1} iterations")
    print("Version 1 centroids:\n", centroids_v1)
    print("Version 1 cluster sizes:", dict(zip(*np.unique(labels_v1, return_counts=True))))

    print(f"\nK-Means Version 2: converged in {iters_v2} iterations")
    print("Version 2 centroids:\n", centroids_v2)
    print("Version 2 cluster sizes:", dict(zip(*np.unique(labels_v2, return_counts=True))))


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Run the main analysis
    main()

    # Run all unit tests
    print("\n" + "=" * 60)
    print("RUNNING UNIT TESTS")
    print("=" * 60)
    unittest.main(argv=[''], verbosity=2, exit=False)