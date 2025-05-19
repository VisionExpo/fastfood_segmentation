# src/visualization/plot_clusters.py

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def plot_elbow(X, max_k=10, save_path="outputs/elbow_plot.png"):
    """
    Plots the elbow curve to help determine the optimal number of clusters.

    Args:
        X (np.ndarray): Scaled feature matrix.
        max_k (int): Maximum number of clusters to test.
        save_path (str): Path to save the elbow plot.
    """
    distortions = []
    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(X)
        distortions.append(model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), distortions, marker='o')
    plt.title("Elbow Method For Optimal k")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia (Distortion)")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Elbow plot saved to: {save_path}")

def plot_clusters(X, labels, save_path="outputs/cluster_plot.png"):
    """
    Visualizes clusters in 2D using PCA.

    Args:
        X (np.ndarray): Scaled feature matrix.
        labels (np.ndarray): Cluster labels for each sample.
        save_path (str): Path to save the cluster plot.
    """
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis', s=50)
    plt.title("Customer Segments (PCA Visualization)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Cluster plot saved to: {save_path}")
