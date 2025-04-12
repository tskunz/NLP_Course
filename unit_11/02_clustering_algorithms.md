# Document Clustering Algorithms

## 1. K-Means Clustering

### Implementation
```python
from typing import List, Dict, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

class KMeansDocumentClustering:
    def __init__(self, n_clusters: int = 5, max_iter: int = 300):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            init='k-means++',
            random_state=42
        )
        
    def fit_predict(self, document_vectors: np.ndarray) -> np.ndarray:
        """Cluster documents using K-means"""
        normalized_vectors = normalize(document_vectors)
        return self.kmeans.fit_predict(normalized_vectors)
    
    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centroids"""
        return self.kmeans.cluster_centers_
    
    def get_cluster_sizes(self, labels: np.ndarray) -> Dict[int, int]:
        """Get number of documents in each cluster"""
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))
```

### Usage Example
```python
# Example usage of K-means clustering
kmeans_clusterer = KMeansDocumentClustering(n_clusters=5)
cluster_labels = kmeans_clusterer.fit_predict(document_vectors)
cluster_sizes = kmeans_clusterer.get_cluster_sizes(cluster_labels)
print("Cluster sizes:", cluster_sizes)
```

## 2. Hierarchical Clustering

### Implementation
```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

class HierarchicalDocumentClustering:
    def __init__(self, n_clusters: int = 5, linkage: str = 'ward'):
        self.n_clusters = n_clusters
        self.clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
        
    def fit_predict(self, document_vectors: np.ndarray) -> np.ndarray:
        """Cluster documents using hierarchical clustering"""
        return self.clustering.fit_predict(document_vectors)
    
    def plot_dendrogram(self, document_vectors: np.ndarray, 
                       max_display: int = 30) -> None:
        """Plot dendrogram of hierarchical clustering"""
        plt.figure(figsize=(10, 7))
        linkage_matrix = linkage(document_vectors, method='ward')
        dendrogram(linkage_matrix, truncate_mode='lastp', p=max_display)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample index or (cluster size)')
        plt.ylabel('Distance')
        plt.show()
```

## 3. DBSCAN Clustering

### Implementation
```python
from sklearn.cluster import DBSCAN
from collections import Counter

class DBSCANDocumentClustering:
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='cosine'
        )
        
    def fit_predict(self, document_vectors: np.ndarray) -> np.ndarray:
        """Cluster documents using DBSCAN"""
        return self.dbscan.fit_predict(document_vectors)
    
    def get_cluster_stats(self, labels: np.ndarray) -> Dict[str, int]:
        """Get clustering statistics"""
        label_counts = Counter(labels)
        stats = {
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'n_noise': list(labels).count(-1)
        }
        return stats
```

## 4. Topic-based Clustering

### Implementation
```python
from sklearn.decomposition import LatentDirichletAllocation
from typing import List, Dict

class TopicBasedClustering:
    def __init__(self, n_topics: int = 5, max_iter: int = 10):
        self.n_topics = n_topics
        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=max_iter,
            learning_method='online',
            random_state=42
        )
        
    def fit_transform(self, document_vectors: np.ndarray) -> np.ndarray:
        """Transform documents to topic space"""
        return self.lda.fit_transform(document_vectors)
    
    def get_topic_terms(self, feature_names: List[str], 
                       top_n: int = 10) -> List[List[str]]:
        """Get top terms for each topic"""
        topics = []
        for topic_idx, topic in enumerate(self.lda.components_):
            top_terms = [feature_names[i] 
                        for i in topic.argsort()[:-top_n-1:-1]]
            topics.append(top_terms)
        return topics
```

## 5. Ensemble Clustering

### Implementation
```python
from typing import List
from sklearn.base import BaseEstimator
import numpy as np

class EnsembleClusterer:
    def __init__(self, base_clusterers: List[BaseEstimator]):
        self.base_clusterers = base_clusterers
        
    def fit_predict(self, document_vectors: np.ndarray) -> np.ndarray:
        """Perform ensemble clustering"""
        predictions = []
        for clusterer in self.base_clusterers:
            pred = clusterer.fit_predict(document_vectors)
            predictions.append(pred)
            
        # Simple majority voting
        ensemble_pred = np.zeros(len(document_vectors))
        for i in range(len(document_vectors)):
            votes = [pred[i] for pred in predictions]
            ensemble_pred[i] = max(set(votes), key=votes.count)
            
        return ensemble_pred
    
    def get_agreement_score(self, predictions: List[np.ndarray]) -> float:
        """Calculate agreement between clusterers"""
        from sklearn.metrics import adjusted_rand_score
        scores = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                score = adjusted_rand_score(predictions[i], predictions[j])
                scores.append(score)
        return np.mean(scores)
```

## Clustering Pipeline

### Implementation
```python
class DocumentClusteringPipeline:
    def __init__(self, vectorizer, clusterer, evaluator):
        self.vectorizer = vectorizer
        self.clusterer = clusterer
        self.evaluator = evaluator
        
    def process_documents(self, documents: List[str]) -> Dict:
        """Run complete clustering pipeline"""
        # Vectorize documents
        vectors = self.vectorizer.vectorize_documents(documents)
        
        # Perform clustering
        labels = self.clusterer.fit_predict(vectors)
        
        # Evaluate clustering
        evaluation = {
            'silhouette': self.evaluator.silhouette_score(vectors, labels),
            'calinski_harabasz': self.evaluator.calinski_harabasz_score(
                vectors, labels
            )
        }
        
        return {
            'vectors': vectors,
            'labels': labels,
            'evaluation': evaluation
        }
```

## Best Practices for Algorithm Selection

### 1. K-Means
- Best for: Well-separated, spherical clusters
- Considerations:
  - Number of clusters known
  - Computationally efficient
  - Sensitive to outliers

### 2. Hierarchical
- Best for: Nested cluster structure
- Considerations:
  - Provides cluster hierarchy
  - Memory intensive
  - Good for small to medium datasets

### 3. DBSCAN
- Best for: Arbitrary shaped clusters
- Considerations:
  - Handles noise
  - No predefined clusters needed
  - Parameter selection critical

### 4. Topic-based
- Best for: Text document collections
- Considerations:
  - Interpretable topics
  - Soft clustering
  - Computationally intensive

### 5. Ensemble
- Best for: Robust clustering
- Considerations:
  - Combines multiple approaches
  - More stable results
  - Higher computational cost

## References
1. Jain, A.K. "Data Clustering: 50 Years Beyond K-means"
2. Ester, M. et al. "A Density-Based Algorithm for Discovering Clusters"
3. Blei, D.M. "Probabilistic Topic Models"

---
*Note: The implementations provided are for educational purposes. Production use may require additional optimization and error handling.* 