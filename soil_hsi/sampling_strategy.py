import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def suggest_sampling_locations(var_map: np.ndarray, num_samples: int, field_acres: float):
    """SoilOptix-style: high/low variability zones + spatial spread."""
    flat_var = var_map.flatten()
    scaler = StandardScaler()
    features = scaler.fit_transform(flat_var.reshape(-1, 1))
    
    k = min(num_samples, 10)  # up to 10 clusters for diversity
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(features)
    labels = kmeans.labels_.reshape(var_map.shape)
    
    # Pick one point per cluster, biased toward high/low variance extremes
    sample_coords = []
    for c in range(k):
        mask = labels == c
        candidates = np.argwhere(mask)
        if len(candidates) == 0:
            continue
        # prefer extremes
        idx = np.random.choice(len(candidates))
        sample_coords.append(candidates[idx])
    
    # Ensure minimum samples
    while len(sample_coords) < max(CFG["min_samples"], int(field_acres * CFG["sampling_ratio"])):
        extra = np.unravel_index(np.random.choice(var_map.size), var_map.shape)
        sample_coords.append(extra)
    
    return np.array(sample_coords)  # (N, 2) row-col indices
