"""
Fast Food Market Segmentation (McDonald's Case Study Replication)
Author: Vishal Gorule
Description: End-to-end segmentation pipeline using clustering.
"""

from src.data.load_data import load_fastfood_data
from src.preprocessing.clean_data import preprocess_data
from src.model.clustering import perform_clustering
from src.visualization.plot_clusters import plot_elbow, plot_clusters


def main():
    # Step 1: Load dataset
    df = load_fastfood_data("data\mcdonalds.csv")

    # Step 2: Preprocess
    X = preprocess_data(df)

    # Step 3: Visualize elbow method
    plot_elbow(X)

    # Step 4: Perform clustering
    model, labels = perform_clustering(X, n_clusters=3)

    # Step 5: Plot cluster visualization
    plot_clusters(X, labels)


if __name__ == "__main__":
    main()
