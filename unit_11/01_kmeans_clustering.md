# K-Means Clustering for Document Analysis

## Introduction to K-Means
K-means clustering is one of the most popular and straightforward clustering algorithms used in document analysis. It partitions n documents into k clusters where each document belongs to the cluster with the nearest mean.

## Algorithm Overview

### Basic Concept
```python
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class DocumentKMeans:
    def __init__(self, n_clusters=5, random_state=42):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english'
        )
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state
        )
        
    def fit_transform(self, documents):
        """Transform documents and fit K-means"""
        vectors = self.vectorizer.fit_transform(documents)
        return self.kmeans.fit_predict(vectors)
```

## Implementation Details

### 1. Document Vectorization
```python
def vectorize_documents(documents):
    """Convert documents to TF-IDF vectors"""
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    return vectorizer.fit_transform(documents)
```

### 2. Centroid Initialization
```python
def initialize_centroids(vectors, n_clusters):
    """Initialize cluster centroids using k-means++"""
    n_samples, n_features = vectors.shape
    centroids = np.zeros((n_clusters, n_features))
    
    # Choose first centroid randomly
    first_centroid = vectors[np.random.randint(n_samples)].toarray()
    centroids[0] = first_centroid
    
    # Choose remaining centroids
    for i in range(1, n_clusters):
        distances = np.min([
            np.linalg.norm(vectors - centroid, axis=1)
            for centroid in centroids[:i]
        ], axis=0)
        probabilities = distances / distances.sum()
        next_centroid_idx = np.random.choice(
            range(n_samples),
            p=probabilities
        )
        centroids[i] = vectors[next_centroid_idx].toarray()
    
    return centroids
```

### 3. Distance Calculation
```python
def calculate_distances(vectors, centroids):
    """Calculate distances between vectors and centroids"""
    distances = np.zeros((vectors.shape[0], len(centroids)))
    for i, centroid in enumerate(centroids):
        distances[:, i] = np.linalg.norm(
            vectors - centroid,
            axis=1
        )
    return distances
```

### 4. Cluster Assignment
```python
def assign_clusters(distances):
    """Assign vectors to nearest centroid"""
    return np.argmin(distances, axis=1)
```

### 5. Centroid Update
```python
def update_centroids(vectors, labels, n_clusters):
    """Update centroid positions"""
    centroids = np.zeros((n_clusters, vectors.shape[1]))
    for i in range(n_clusters):
        cluster_vectors = vectors[labels == i]
        if len(cluster_vectors) > 0:
            centroids[i] = cluster_vectors.mean(axis=0)
    return centroids
```

## Advanced Features

### 1. Multiple Initializations
```python
def kmeans_multiple_init(vectors, n_clusters, n_init=10):
    """Run K-means with multiple initializations"""
    best_inertia = float('inf')
    best_labels = None
    
    for _ in range(n_init):
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=np.random.randint(1000)
        )
        labels = kmeans.fit_predict(vectors)
        
        if kmeans.inertia_ < best_inertia:
            best_inertia = kmeans.inertia_
            best_labels = labels
    
    return best_labels
```

### 2. Elbow Method
```python
def find_optimal_k(vectors, max_k=20):
    """Find optimal number of clusters using elbow method"""
    inertias = []
    k_values = range(1, max_k + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(vectors)
        inertias.append(kmeans.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.show()
    
    return inertias
```

## Best Practices

### 1. Data Preprocessing
- Remove stop words and rare terms
- Apply text normalization
- Consider using n-grams
- Scale feature vectors

### 2. Parameter Selection
- Choose appropriate number of clusters
- Use multiple initializations
- Consider feature dimensionality
- Balance computational cost

### 3. Evaluation
- Use silhouette score
- Check cluster sizes
- Validate cluster coherence
- Compare with other methods

## Applications

### 1. Document Organization
- Automatic document categorization
- Content recommendation
- Search result clustering
- Topic discovery

### 2. Feature Learning
- Document representation
- Semantic analysis
- Dimensionality reduction
- Pattern discovery

## Limitations and Considerations

### 1. Algorithm Limitations
- Assumes spherical clusters
- Sensitive to initialization
- Requires predefined k
- May converge to local optima

### 2. Document-Specific Issues
- High dimensionality
- Sparse vectors
- Term ambiguity
- Cluster interpretation

## References
1. MacQueen, J. "Some Methods for Classification and Analysis of Multivariate Observations"
2. Arthur, D. and Vassilvitskii, S. "k-means++: The Advantages of Careful Seeding"
3. Manning, C.D. et al. "Introduction to Information Retrieval"

---
*Note: The implementations provided are for educational purposes. Production use may require additional optimization and error handling.* 