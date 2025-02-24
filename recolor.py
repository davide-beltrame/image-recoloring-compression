#!/usr/bin/env python3
import sys
import numpy as np
import random
import time
from PIL import Image

start = time.time()

def compute_distances(pixels, centres):
    diff = pixels[:, np.newaxis, :] - centres[np.newaxis, :, :]
    distances = np.sum(diff ** 2, axis=2)  # squared distances
    return distances

def initialise_centres_kmeans_pp(pixels, k):
    num_pixels = pixels.shape[0]
    centres = []
    first_centre_index = np.random.randint(0, num_pixels)
    centres.append(pixels[first_centre_index])

    for _ in range(1, k):
        distances = compute_distances(pixels, np.array(centres))
        min_distances = np.min(distances, axis=1)
        total_distance = np.sum(min_distances)
        if total_distance == 0: # all points are identical
            centres.append(pixels[np.random.randint(0, num_pixels)])
            continue
        probabilities = min_distances / total_distance
        cumulative_probabilities = np.cumsum(probabilities)
        r = np.random.rand()
        next_centre_index = np.searchsorted(cumulative_probabilities, r)
        centres.append(pixels[next_centre_index])
    return np.array(centres, dtype=float)

def kmeans(pixels, k, max_iterations=300):
    num_pixels = pixels.shape[0]
    centres = initialise_centres_kmeans_pp(pixels, k)
    
    for _ in range(max_iterations):
        distances = compute_distances(pixels, centres)
        labels = np.argmin(distances, axis=1)  # assign each pixel to the closest centre
        new_centres = np.array([
            pixels[labels == i].mean(axis=0) if np.any(labels == i) else centres[i]
            for i in range(k)
        ])  # recompute centres
        
        if np.allclose(centres, new_centres, atol=1e-4):
            break
        centres = new_centres
    min_distances = distances[np.arange(num_pixels), labels]  # distance of each pixel to its assigned centre
    total_cost = np.sum(min_distances)
    normalised_total_cost = total_cost / num_pixels
    return centres, labels, total_cost, normalised_total_cost

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 recolor.py {input-image} {output-image} {k}")
        sys.exit(1)
    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]
    k = sys.argv[3]
    try:
        k = int(k)
        if k < 1:
            print("Error: k must be a positive integer.")
            sys.exit(1)
    except ValueError:
        print("Error: k must be a positive integer.")
        sys.exit(1)
    try:
        image = Image.open(input_image_path)
    except FileNotFoundError:
        print(f"Error: Input image '{input_image_path}' not found.")
        sys.exit(1)
    except IOError:
        print(f"Error: Cannot open '{input_image_path}'. Ensure it's a valid image file.")
        sys.exit(1)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_data = np.array(image)
    pixels = image_data.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)
    num_unique_colors = unique_colors.shape[0]
    if k >= num_unique_colors:
        print(f"The image has only {num_unique_colors} unique colors. Saving the original image.")
        image.save(output_image_path)
        sys.exit(0)
    num_runs = 30
    best_total_cost = None
    best_normalised_total_cost = None
    best_centres = None
    best_labels = None
    print(f"Running k-means clustering {num_runs} times to find the best result...")
    for run in range(num_runs):
        random_seed = random.randint(0, int(1e9))
        random.seed(random_seed)
        np.random.seed(random_seed)
        centres, labels, total_cost, normalised_total_cost = kmeans(pixels, k)
        print(f"Run {run + 1}/{num_runs}, TC: {total_cost:.2f}, NTC: {normalised_total_cost:.2f}")
        if best_total_cost is None or total_cost < best_total_cost:
            best_total_cost = total_cost
            best_normalised_total_cost = normalised_total_cost
            best_centres = centres
            best_labels = labels
    new_pixels = best_centres[best_labels].astype(np.uint8)
    new_image_data = new_pixels.reshape(image_data.shape)
    new_image = Image.fromarray(new_image_data, mode='RGB')
    new_image.save(output_image_path)
    print(f"\nBest result with Total Cost: {best_total_cost:.2f}")
    print(f"Best result with Normalized Total Cost: {best_normalised_total_cost:.2f}")
    print(f"Output image saved to '{output_image_path}' with {k} colors.")
    
    with open('best_total_cost.txt', 'w') as f:
        f.write(f"{best_total_cost}\n")
    print(f"Best total cost saved to 'best_total_cost.txt'.")
    np.save('best_centres.npy', best_centres)
    print(f"Best centres saved to 'best_centres.npy'.")
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds.")

if __name__ == "__main__":
    main()