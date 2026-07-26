# A1 [ID - Nominal, Year_Birth - Ratio, Education - Ordinal, Marital_Status - Nominal
# Income - Ratio, Kidhome - Ratio, Teenhome - Ratio, Dt_Customer - Interval]
import numpy as n
import pandas as p
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski

# A2
def load_data(file):
    data = p.read_excel(file, sheet_name="marketing_campaign")
    return data

def label_encoding(column):
    encoded_column = {}
    unique_values = column.unique()
    for i, value in enumerate(unique_values):
        encoded_column[value] = i
    encoded_data = column.map(encoded_column)
    return encoded_data

def one_hot_encoding(column):
    encoded_data = p.get_dummies(column, dtype=int)
    return encoded_data

# A3
def recreate_dataset_label(data):
    encoded_data = data.copy()
    encoded_data["Education"] = label_encoding(data["Education"])
    encoded_data["Marital_Status"] = label_encoding(data["Marital_Status"])
    return encoded_data

def recreate_dataset_one_hot(data):
    encoded_data = data.copy()
    education_encoded = one_hot_encoding(data["Education"])
    marital_encoded = one_hot_encoding(data["Marital_Status"])
    encoded_data = encoded_data.drop(["Education", "Marital_Status"],axis=1)
    encoded_data = p.concat([encoded_data,education_encoded,marital_encoded],axis=1)
    return encoded_data

def feature_dimension(data):
    rows, columns = data.shape
    return rows, columns

# A4
def minkowski_distance(vector1, vector2, p):
    distance = 0
    for i in range(len(vector1)):
        distance += abs(vector1[i] - vector2[i]) ** p
    distance = distance ** (1 / p)
    return distance

# A5
def minkowski_plot(vector1, vector2):
    p_values = []
    distance_values = []
    for p in range(1, 11):
        distance = minkowski_distance(vector1,vector2,p)
        p_values.append(p)
        distance_values.append(distance)
    plt.plot(p_values,distance_values,marker="o")
    plt.xlabel("Value of p")
    plt.ylabel("Minkowski Distance")
    plt.title("Minkowski Distance vs p")
    plt.show()
    return distance_values

# A6
def scipy_minkowski_distance(vector1, vector2, p):
    distance = minkowski(vector1,vector2,p)
    return distance

# A7
def dot_product(vector1, vector2):
    dot = 0
    for i in range(len(vector1)):
        dot += vector1[i] * vector2[i]
    return dot

def euclidean_norm(vector):
    total = 0
    for value in vector:
        total += value ** 2
    norm = total ** 0.5
    return norm

# A8
def mean(data):
    total = 0
    for value in data:
        total += value
    return total / len(data)

def variance(data):
    mean_value = mean(data)
    total = 0
    for value in data:
        total += (value - mean_value) ** 2
    return total / len(data)

def standard_deviation(data):
    variance_value = variance(data)
    return variance_value ** 0.5

def dataset_statistics(data):
    numerical_data = data.select_dtypes(
        include=["int64", "float64"]
    )
    statistics = {}
    for column in numerical_data.columns:
        values = numerical_data[column].values
        statistics[column] = {
            "Mean": mean(values),
            "Variance": variance(values),
            "Standard Deviation":
                standard_deviation(values)
        }
    return statistics

# A9
def numpy_statistics(data):
    numerical_data = data.select_dtypes(
        include=["int64", "float64"]
    )
    statistics = {}
    for column in numerical_data.columns:
        values = numerical_data[column].values
        statistics[column] = {
            "Mean":
                n.mean(values),
            "Standard Deviation":
                n.std(values)
        }
    return statistics

# A10
def histogram(feature):
    mean_value = mean(feature)
    variance_value = variance(feature)
    plt.hist(feature,bins=10)
    plt.xlabel("Feature Values")
    plt.ylabel("Frequency")
    plt.title("Histogram")
    plt.show()
    return mean_value, variance_value

# A11
def assign_clusters(data, centroids):
    clusters = []
    for point in data:
        distances = []
        for centroid in centroids:
            distance = minkowski_distance(point,centroid,2)
            distances.append(distance)
        cluster = distances.index(min(distances))
        clusters.append(cluster)
    return n.array(clusters)

def calculate_centroids(data,clusters,k):
    centroids = []
    for i in range(k):
        cluster_points = data[clusters == i]
        centroid = n.mean(cluster_points,axis=0)
        centroids.append(centroid)
    return n.array(centroids)

def k_means(data,k,iterations=100):
    centroids = data[n.random.choice(len(data),k,replace=False)]
    for _ in range(iterations):
        clusters = assign_clusters(data,centroids)
        new_centroids = calculate_centroids(data,clusters,k)
        if n.allclose(centroids,new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids

def main():
    file = "Lab Session Data.xlsx"
    data = load_data(file)
    education = data["Education"]
    marital = data["Marital_Status"]

    # A2
    print("EDUCATION COLUMN\n")
    print(education)
    print("\nLabel Encoding of Education\n")
    print(label_encoding(education))
    print("\nOne Hot Encoding of Education\n")
    print(one_hot_encoding(education))
    print("\n\nMARITAL STATUS COLUMN\n")
    print(marital)
    print("\nLabel Encoding of Marital Status\n")
    print(label_encoding(marital))
    print("\nOne Hot Encoding of Marital Status\n")
    print(one_hot_encoding(marital))

    # A3
    rows, columns = feature_dimension(data)
    print("\nOriginal Dataset Dimension =",rows, "x", columns)
    label_dataset = recreate_dataset_label(data)
    rows, columns = feature_dimension(label_dataset)
    print("\nLabel Encoded Dataset Dimension =",rows, "x", columns)
    one_hot_dataset = recreate_dataset_one_hot(data)
    rows, columns = feature_dimension(one_hot_dataset)
    print("\nOne Hot Encoded Dataset Dimension =",rows, "x", columns)

    # A4, A5 and A6
    numerical_data = data.select_dtypes(include=["int64", "float64"])
    vector1 = numerical_data.iloc[0].values
    vector2 = numerical_data.iloc[1].values

    # A4
    print("\nManhattan Distance =",minkowski_distance(vector1,vector2,1))
    print("\nEuclidean Distance =",minkowski_distance(vector1,vector2,2))

    # A5
    minkowski_plot(vector1,vector2)

    # A6
    for p in range(1, 11):
        own_distance = minkowski_distance(vector1,vector2,p)
        scipy_distance = scipy_minkowski_distance(vector1,vector2,p)
        print("\np =", p)
        print("Own Function =",own_distance)
        print("Scipy Function =",scipy_distance)

    # A7
    own_dot_product = dot_product(vector1,vector2)
    numpy_dot_product = n.dot(vector1,vector2)
    print("\nOwn Dot Product =",own_dot_product)
    print("NumPy Dot Product =",numpy_dot_product)
    own_norm_vector1 = euclidean_norm(vector1)
    numpy_norm_vector1 = n.linalg.norm(vector1)
    print("\nOwn Euclidean Norm =",own_norm_vector1)
    print("NumPy Euclidean Norm =",numpy_norm_vector1)
    own_norm_vector2 = euclidean_norm(vector2)
    numpy_norm_vector2 = n.linalg.norm(vector2)
    print("\nOwn Euclidean Norm =",own_norm_vector2)
    print("NumPy Euclidean Norm =",numpy_norm_vector2)   

    # A8
    statistics = dataset_statistics(data)
    print("\nA8 Results\n")
    for column, values in statistics.items():
        print("\n", column)
        print("Mean =",values["Mean"])
        print("Variance =",values["Variance"])
        print("Standard Deviation =",values["Standard Deviation"])

# A9
    numpy_values = numpy_statistics(data)
    print("\nA9 Results\n")
    for column, values in numpy_values.items():
        print("\n", column)
    print("Mean =",values["Mean"])
    print("Standard Deviation =",values["Standard Deviation"]) 

# A10
    income = data["Income"].dropna().values
    mean_value, variance_value = histogram(income)
    print("\nIncome Feature")
    print("Mean =",mean_value)
    print("Variance =",variance_value)

# A11
    numerical_data = data.select_dtypes(
    include=["int64", "float64"])
    kmeans_data = numerical_data.fillna(
    numerical_data.mean()
    ).values
    clusters, centroids = k_means(
    kmeans_data,
    k=3
)
    print("\nClusters =")
    print(clusters)
    print("\nCentroids =")
    print(centroids)
    
if __name__ == "__main__":
    main()