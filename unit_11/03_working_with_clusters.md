# Working with Document Clusters

## Cluster Analysis and Interpretation

### 1. Cluster Summarization
```python
from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class ClusterSummarizer:
    def __init__(self, n_terms: int = 10):
        self.n_terms = n_terms
        
    def get_cluster_terms(self, documents: List[str], 
                         labels: np.ndarray) -> Dict[int, List[str]]:
        """Extract representative terms for each cluster"""
        vectorizer = TfidfVectorizer(max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
        
        cluster_terms = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label != -1:  # Exclude noise points
                cluster_docs = tfidf_matrix[labels == label]
                centroid = cluster_docs.mean(axis=0).A1
                top_indices = centroid.argsort()[-self.n_terms:][::-1]
                cluster_terms[label] = [feature_names[i] for i in top_indices]
        
        return cluster_terms
    
    def get_cluster_documents(self, documents: List[str], 
                            labels: np.ndarray,
                            n_docs: int = 5) -> Dict[int, List[str]]:
        """Get representative documents for each cluster"""
        unique_labels = np.unique(labels)
        cluster_docs = {}
        
        for label in unique_labels:
            if label != -1:
                cluster_indices = np.where(labels == label)[0]
                # Get first n_docs as representatives (can be improved)
                docs = [documents[i] for i in cluster_indices[:n_docs]]
                cluster_docs[label] = docs
        
        return cluster_docs
```

### 2. Cluster Visualization
```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class ClusterVisualizer:
    def __init__(self):
        self.tsne = TSNE(n_components=2, random_state=42)
        
    def plot_clusters(self, vectors: np.ndarray, 
                     labels: np.ndarray,
                     title: str = "Document Clusters") -> None:
        """Visualize clusters in 2D space"""
        # Reduce dimensions for visualization
        vectors_2d = self.tsne.fit_transform(vectors.toarray())
        
        # Plot clusters
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], 
                            c=labels, cmap='viridis')
        plt.colorbar(scatter)
        plt.title(title)
        plt.xlabel("t-SNE dimension 1")
        plt.ylabel("t-SNE dimension 2")
        plt.show()
    
    def plot_cluster_sizes(self, labels: np.ndarray) -> None:
        """Visualize cluster size distribution"""
        unique, counts = np.unique(labels, return_counts=True)
        plt.figure(figsize=(10, 6))
        plt.bar(unique, counts)
        plt.title("Cluster Size Distribution")
        plt.xlabel("Cluster Label")
        plt.ylabel("Number of Documents")
        plt.show()
```

## Cluster Refinement

### 1. Cluster Quality Assessment
```python
from sklearn.metrics import silhouette_samples

class ClusterQualityAnalyzer:
    def analyze_cluster_quality(self, vectors: np.ndarray, 
                              labels: np.ndarray) -> Dict[str, float]:
        """Analyze quality of clustering results"""
        # Calculate silhouette scores for each sample
        silhouette_vals = silhouette_samples(vectors, labels)
        
        # Analyze cluster cohesion and separation
        cluster_scores = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label != -1:
                cluster_silhouette = silhouette_vals[labels == label]
                cluster_scores[f'cluster_{label}'] = {
                    'mean_silhouette': np.mean(cluster_silhouette),
                    'std_silhouette': np.std(cluster_silhouette),
                    'size': np.sum(labels == label)
                }
        
        return cluster_scores
    
    def identify_outliers(self, vectors: np.ndarray, 
                         labels: np.ndarray,
                         threshold: float = 0.1) -> List[int]:
        """Identify potential outliers based on silhouette scores"""
        silhouette_vals = silhouette_samples(vectors, labels)
        return np.where(silhouette_vals < threshold)[0].tolist()
```

### 2. Cluster Optimization
```python
class ClusterOptimizer:
    def __init__(self, min_clusters: int = 2, max_clusters: int = 20):
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        
    def find_optimal_clusters(self, vectors: np.ndarray) -> Dict[str, any]:
        """Find optimal number of clusters using multiple metrics"""
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        
        scores = {
            'n_clusters': [],
            'silhouette': [],
            'calinski_harabasz': []
        }
        
        for n in range(self.min_clusters, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=n, random_state=42)
            labels = kmeans.fit_predict(vectors)
            
            scores['n_clusters'].append(n)
            scores['silhouette'].append(
                silhouette_score(vectors, labels)
            )
            scores['calinski_harabasz'].append(
                calinski_harabasz_score(vectors, labels)
            )
        
        return scores
```

## Practical Applications

### 1. Document Organization
```python
class DocumentOrganizer:
    def __init__(self, base_path: str):
        self.base_path = base_path
        
    def organize_documents(self, documents: List[str], 
                         labels: np.ndarray,
                         filenames: List[str]) -> Dict[int, List[str]]:
        """Organize documents into folders based on clusters"""
        import os
        
        organization = {}
        for doc, label, filename in zip(documents, labels, filenames):
            if label != -1:
                cluster_dir = os.path.join(self.base_path, f'cluster_{label}')
                os.makedirs(cluster_dir, exist_ok=True)
                
                # Save document to cluster directory
                with open(os.path.join(cluster_dir, filename), 'w') as f:
                    f.write(doc)
                
                if label not in organization:
                    organization[label] = []
                organization[label].append(filename)
        
        return organization
```

### 2. Search Enhancement
```python
class ClusterBasedSearch:
    def __init__(self, vectorizer, clusterer):
        self.vectorizer = vectorizer
        self.clusterer = clusterer
        self.document_vectors = None
        self.labels = None
        
    def index_documents(self, documents: List[str]) -> None:
        """Index documents for cluster-based search"""
        self.document_vectors = self.vectorizer.vectorize_documents(documents)
        self.labels = self.clusterer.fit_predict(self.document_vectors)
        
    def search(self, query: str, n_results: int = 5) -> List[int]:
        """Search for relevant documents using cluster information"""
        # Vectorize query
        query_vector = self.vectorizer.vectorize_documents([query])
        
        # Find closest cluster
        cluster_centers = self.clusterer.get_cluster_centers()
        closest_cluster = np.argmin(
            [np.linalg.norm(query_vector - center) 
             for center in cluster_centers]
        )
        
        # Get documents from closest cluster
        cluster_docs = np.where(self.labels == closest_cluster)[0]
        
        # Rank documents within cluster
        similarities = [
            np.dot(query_vector.toarray()[0], 
                  self.document_vectors[i].toarray()[0])
            for i in cluster_docs
        ]
        
        # Return top n results
        top_indices = np.argsort(similarities)[-n_results:][::-1]
        return cluster_docs[top_indices].tolist()
```

## Best Practices

### 1. Data Management
- Regular cluster maintenance
- Version control for cluster models
- Document metadata tracking

### 2. Performance Optimization
- Incremental clustering
- Parallel processing
- Caching strategies

### 3. User Interface
- Interactive visualization
- Search integration
- Feedback collection

## Applications

### 1. Content Management
- Document categorization
- Archive organization
- Content recommendation

### 2. Information Retrieval
- Faceted search
- Similar document finding
- Topic browsing

### 3. Knowledge Discovery
- Trend analysis
- Pattern identification
- Content summarization

## References
1. Aggarwal, C.C. & Reddy, C.K. "Data Clustering: Algorithms and Applications"
2. Steinbach, M. et al. "A Comparison of Document Clustering Techniques"
3. Zhao, Y. & Karypis, G. "Evaluation of Hierarchical Clustering Algorithms for Document Datasets"

---
*Note: The implementations provided are for educational purposes. Production use may require additional optimization and error handling.* 