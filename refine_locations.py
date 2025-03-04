def refine_locations(cluster_centers, existing_businesses, step_size=0.01, max_iterations=10):
    """Moves business locations slightly to increase population coverage and avoid competitors."""
    
    refined_centers = cluster_centers.copy()
    
    for _ in range(max_iterations):
        for i, center in enumerate(refined_centers):
            lat, lon = center

            # Compute distance to closest existing business
            min_distance_to_competitor = min(geodesic((lat, lon), (b[0], b[1])).km for b in existing_businesses)

            # Compute distance to nearest population centroid
            min_distance_to_population = min(geodesic((lat, lon), (p[0], p[1])).km for p in coords)

            # Move center slightly in the direction maximizing distance to competitors
            if min_distance_to_competitor < 2.0:  # If too close to a competitor, move
                lat += np.random.uniform(-step_size, step_size)
                lon += np.random.uniform(-step_size, step_size)

            # Keep the location close to population clusters
            if min_distance_to_population > 5.0:  
                lat -= np.random.uniform(-step_size, step_size)
                lon -= np.random.uniform(-step_size, step_size)

            refined_centers[i] = [lat, lon]

    return np.array(refined_centers)

# Apply local search refinement
optimized_locations = refine_locations(cluster_centers, existing_businesses)

# Plot final results
plt.scatter(coords[:, 1], coords[:, 0], c='blue', alpha=0.3, label="Population Centroids")
plt.scatter(existing_businesses[:, 1], existing_businesses[:, 0], c='red', label="Existing Businesses")
plt.scatter(optimized_locations[:, 1], optimized_locations[:, 0], c='green', marker='x', label="Optimized New Locations")
plt.legend()
plt.show()
