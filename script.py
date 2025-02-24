#!/usr/bin/env python3
import numpy as np
from PIL import Image
import sys
import itertools

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 script.py {input-image}")
        sys.exit(1)
    input_image_path = sys.argv[1]
    try:
        image = Image.open(input_image_path)
    except (FileNotFoundError, IOError):
        print(f"Error: Cannot open '{input_image_path}'. Ensure it's a valid image file.")
        sys.exit(1)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_data = np.array(image)
    pixels = image_data.reshape(-1, 3)
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    num_unique_colors = unique_colors.shape[0]
    print(f"Number of unique colors: {num_unique_colors}")

    try:
        with open('best_total_cost.txt', 'r') as f:
            best_total_cost = float(f.readline().strip())
    except (FileNotFoundError, ValueError):
        print("Error: 'best_total_cost.txt' not found or invalid. Please run recolor.py first.")
        sys.exit(1)

    color_indices = list(range(num_unique_colors))
    color_weights = counts
    color_values = unique_colors
    max_cluster_size = 3
    # adjust the previous variable as needed, 
    # a value of 3 assumes decently spread colours for 8 centroids in our 20 colour palette,
    # a value of 20 instead relax completely the constraint and will generate all possible clusters,
    # but the result will be the same
    clusters = []
    print("Generating possible clusters...")

    for size in range(1, max_cluster_size + 1):
        combinations = itertools.combinations(color_indices, size)
        for combo in combinations:
            cluster_indices = list(combo)
            cluster_colors = color_values[cluster_indices]
            cluster_weights = color_weights[cluster_indices]
            total_weight = np.sum(cluster_weights)
            centroid = np.average(cluster_colors, axis=0, weights=cluster_weights)

            diffs = cluster_colors - centroid
            squared_diffs = np.sum(diffs ** 2, axis=1)
            total_cost = np.sum(cluster_weights * squared_diffs)
            if total_cost <= best_total_cost:
                clusters.append({
                    'indices': cluster_indices,
                    'centroid': centroid,
                    'total_cost': total_cost
                })
    print(f"Number of clusters generated: {len(clusters)}")

    if len(clusters) < 8:
        print("Not enough clusters generated. Try increasing 'max_cluster_size' or review your filtering criteria.")
        sys.exit(1)
    n = num_unique_colors  # Number of data points (unique colours)
    P = len(clusters)      # Number of possible clusters
    k = 8                  # Number of clusters to select
    c = [cluster['total_cost'] for cluster in clusters]
    C = np.zeros((P, n), dtype=int)
    for j, cluster in enumerate(clusters):
        for idx in cluster['indices']:
            C[j][idx] = 1

    with open('k_means.dat', 'w') as f:
        f.write("# Usage: glpsol -m k_means.mod -d k_means.dat\n\n")
        f.write("data;\n\n")
        f.write(f"param n := {n};\n")
        f.write(f"param P := {P};\n")
        f.write(f"param k := {k};\n\n")
        f.write("param c :=\n")
        for j in range(P):
            f.write(f" {j+1} {c[j]}\n")
        f.write(";\n\n")
        f.write("param C : ")
        f.write(' '.join(str(i+1) for i in range(n)) + " :=\n")
        for j in range(P):
            f.write(f" {j+1} " + ' '.join(str(C[j][i]) for i in range(n)) + "\n")
        f.write(";\n\n")
        f.write("end;\n")
    print("File 'k_means.dat' has been generated.")

if __name__ == "__main__":
    main()