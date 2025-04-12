# Hierarchical Document Clustering

## Introduction to Hierarchical Clustering
Hierarchical clustering creates a tree-like structure of document relationships, allowing for multi-level analysis of document similarities. This approach is particularly useful when the natural number of clusters is unknown or when a hierarchical organization is desired.

## Types of Hierarchical Clustering

### 1. Agglomerative (Bottom-up)
```python
from sklearn.cluster import AgglomerativeClustering
import numpy as np

class AgglomerativeDocumentClustering:
    def __init__(self, n_clusters=None, distance_threshold=None):
        self.clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            distance_threshold=distance_threshold,
            linkage='ward'
        )
    
    def fit_predict(self, vectors):
        """Perform hierarchical clustering"""
        return self.clustering.fit_predict(vectors.toarray())
```

### 2. Divisive (Top-down)
```python
class DivisiveDocumentClustering:
    def __init__(self, max_clusters=None):
        self.max_clusters = max_clusters
        
    def split_cluster(self, vectors):
        """Split cluster using K-means with k=2"""
        kmeans = KMeans(n_clusters=2, random_state=42)
        return kmeans.fit_predict(vectors)
    
    def fit_predict(self, vectors):
        """Perform divisive clustering"""
        n_samples = vectors.shape[0]
        labels = np.zeros(n_samples)
        current_label = 0
        
        clusters_to_split = [(vectors, np.arange(n_samples))]
        while clusters_to_split:
            cluster_vectors, cluster_indices = clusters_to_split.pop(0)
            if len(cluster_indices) > 1:
                split_labels = self.split_cluster(cluster_vectors)
                for i in range(2):
                    subset_indices = cluster_indices[split_labels == i]
                    if len(subset_indices) > 1:
                        subset_vectors = vectors[subset_indices]
                        clusters_to_split.append(
                            (subset_vectors, subset_indices)
                        )
                    labels[subset_indices] = current_label
                    current_label += 1
        
        return labels
```

## Implementation Details

### 1. Distance Metrics
```python
def calculate_distance_matrix(vectors, metric='euclidean'):
    """Calculate pairwise distances between documents"""
    from scipy.spatial.distance import pdist, squareform
    
    if metric == 'euclidean':
        distances = pdist(vectors.toarray(), metric='euclidean')
    elif metric == 'cosine':
        distances = pdist(vectors.toarray(), metric='cosine')
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return squareform(distances)
```

### 2. Linkage Methods
```python
def compute_linkage(distance_matrix, method='ward'):
    """Compute linkage matrix for hierarchical clustering"""
    from scipy.cluster.hierarchy import linkage
    
    linkage_matrix = linkage(
        distance_matrix,
        method=method,
        optimal_ordering=True
    )
    return linkage_matrix
```

### 3. Dendrogram Visualization
```python
def plot_dendrogram(linkage_matrix, labels=None, max_d=None):
    """Visualize hierarchical clustering as dendrogram"""
    from scipy.cluster.hierarchy import dendrogram
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    dendrogram(
        linkage_matrix,
        labels=labels,
        color_threshold=max_d,
        leaf_rotation=90
    )
    if max_d:
        plt.axhline(y=max_d, color='r', linestyle='--')
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()
```

## Advanced Features

### 1. Cluster Extraction
```python
def extract_clusters(linkage_matrix, distance_threshold):
    """Extract flat clusters from hierarchical clustering"""
    from scipy.cluster.hierarchy import fcluster
    
    labels = fcluster(
        linkage_matrix,
        distance_threshold,
        criterion='distance'
    )
    return labels
```

### 2. Cophenetic Correlation
```python
def compute_cophenetic_correlation(linkage_matrix, distance_matrix):
    """Compute cophenetic correlation coefficient"""
    from scipy.cluster.hierarchy import cophenet
    from scipy.stats import pearsonr
    
    cophenetic_distances = cophenet(linkage_matrix)
    correlation = pearsonr(
        distance_matrix[np.triu_indices(len(distance_matrix), k=1)],
        cophenetic_distances
    )[0]
    return correlation
```

### 3. Inconsistency Analysis
```python
def analyze_inconsistency(linkage_matrix, depth=2):
    """Analyze inconsistency of hierarchical clustering"""
    from scipy.cluster.hierarchy import inconsistent
    
    inconsistency = inconsistent(linkage_matrix, depth)
    return inconsistency
```

## Best Practices

### 1. Algorithm Selection
- Choose appropriate linkage method
- Consider data size and structure
- Balance computational resources
- Evaluate different distance metrics

### 2. Parameter Tuning
- Set appropriate distance threshold
- Choose optimal number of clusters
- Consider depth of analysis
- Validate cluster stability

### 3. Visualization
- Use interactive dendrograms
- Implement zoom capabilities
- Add cluster annotations
- Highlight important clusters

## Applications

### 1. Document Organization
- Create document hierarchies
- Build taxonomies
- Organize digital libraries
- Structure knowledge bases

### 2. Content Analysis
- Topic hierarchy discovery
- Relationship mapping
- Content categorization
- Semantic structure analysis

## Limitations and Considerations

### 1. Computational Complexity
- O(n²) space complexity
- O(n³) time complexity
- Memory constraints
- Scalability issues

### 2. Practical Challenges
- Non-reversible decisions
- Sensitivity to noise
- Interpretation complexity
- Validation difficulty

## Integration with Other Methods

### 1. Hybrid Approaches
```python
class HybridClustering:
    def __init__(self, n_clusters=5):
        self.hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters
        )
        self.kmeans = KMeans(
            n_clusters=n_clusters
        )
    
    def fit_predict(self, vectors):
        """Combine hierarchical and k-means clustering"""
        # Get initial clusters from hierarchical clustering
        hier_labels = self.hierarchical.fit_predict(vectors)
        
        # Refine clusters using k-means
        kmeans_labels = self.kmeans.fit_predict(vectors)
        
        # Combine predictions (example: majority voting)
        from scipy.stats import mode
        final_labels = mode([hier_labels, kmeans_labels], axis=0)[0]
        
        return final_labels
```

### 2. Ensemble Methods
```python
def ensemble_hierarchical(vectors, n_bootstraps=10):
    """Create ensemble of hierarchical clusterings"""
    n_samples = vectors.shape[0]
    ensemble_labels = np.zeros((n_bootstraps, n_samples))
    
    for i in range(n_bootstraps):
        # Bootstrap sample
        indices = np.random.choice(
            n_samples,
            size=n_samples,
            replace=True
        )
        bootstrap_vectors = vectors[indices]
        
        # Perform clustering
        clustering = AgglomerativeClustering(n_clusters=5)
        labels = clustering.fit_predict(bootstrap_vectors)
        
        # Store results
        ensemble_labels[i] = labels
    
    # Combine results (example: majority voting)
    final_labels = mode(ensemble_labels, axis=0)[0]
    
    return final_labels
```

## References
1. Murtagh, F. and Contreras, P. "Algorithms for Hierarchical Clustering: An Overview"
2. Müllner, D. "Modern Hierarchical, Agglomerative Clustering Algorithms"
3. Zhao, Y. and Karypis, G. "Evaluation of Hierarchical Clustering Algorithms for Document Datasets"

---
*Note: The implementations provided are for educational purposes. Production use may require additional optimization and error handling.* 