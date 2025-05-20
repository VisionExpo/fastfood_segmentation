"""
Fast Food Market Segmentation (McDonald's Case Study Replication)
Author: Vishal Gorule
Description: End-to-end segmentation pipeline using clustering.
"""

from src.data.load_data import load_fastfood_data
from src.preprocessing.clean_data import preprocess_data
from src.model.clustering import perform_clustering
from sklearn.cluster import KMeans # For calculating inertias
from kneed import KneeLocator      # For finding the elbow point
from src.visualization.plot_clusters import plot_elbow, plot_clusters


def main():
    # Step 1: Load dataset
    df = load_fastfood_data("data\mcdonalds.csv")

    # Step 2: Preprocess
    X = preprocess_data(df)

    # Step 3: Determine optimal k using Elbow Method programmatically
    print("[INFO] Calculating inertias for elbow method to find optimal k...")
    inertias = []
    # Define a range of k values to test. Adjust max_k as needed.
    # For McDonald's dataset, 1 to 10 is usually sufficient.
    k_range = range(1, 11)
    for k_val in k_range:
        # n_init='auto' is for scikit-learn >= 1.4, use n_init=10 for older versions if you see warnings
        kmeans_model = KMeans(n_clusters=k_val, random_state=42, n_init='auto', algorithm='lloyd')
        kmeans_model.fit(X)
        inertias.append(kmeans_model.inertia_)

    # Use KneeLocator to find the elbow point
    # S=1.0 is a sensitivity parameter, might need tuning for different datasets.
    # curve='convex' and direction='decreasing' are typical for inertia plots.
    kneedle = KneeLocator(list(k_range), inertias, S=1.0, curve='convex', direction='decreasing')
    optimal_k = kneedle.elbow

    if optimal_k is None:
        print("[WARNING] Could not automatically determine optimal k. Defaulting to 3 clusters.")
        optimal_k = 3 # A sensible fallback
    else:
        print(f"[INFO] Optimal number of clusters automatically determined: {optimal_k}")

    # Step 4: Visualize elbow method (for user confirmation/record)
    # This will plot the elbow. It might recalculate inertias internally,
    # or you could modify plot_elbow to accept k_range and inertias.
    plot_elbow(X)

    # Step 5: Perform clustering with the determined optimal_k
    model, labels = perform_clustering(X, n_clusters=optimal_k)

    # Step 6: Plot cluster visualization
    plot_clusters(X, labels)


if __name__ == "__main__":
    main()
