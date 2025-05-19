from sklearn.cluster import KMeans

def perform_clustering(X, n_clusters=3):
    """
    Perform KMeans clustering on the preprocessed data.

    Args:
        X (np.ndarray): Scaled feature matrix.
        n_clusters (int): Number of clusters to form.

    Returns:
        model (KMeans): Trained KMeans model.
        labels (np.ndarray): Cluster labels assigned to each sample.
    """
    print(f"[INFO] Clustering into {n_clusters} segments...")
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X)
    print(f"[INFO] Clustering completed. Inertia: {model.inertia_}")
    return model, labels
