"""
Solutions to Document Clustering Exercises
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import defaultdict

# Exercise 1: Basic Document Clustering
def cluster_documents(documents: List[str], n_clusters: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implement document clustering using TF-IDF and K-means
    
    Args:
        documents: List of text documents
        n_clusters: Number of clusters to create
        
    Returns:
        Tuple of (cluster labels, cluster centers)
    """
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    vectors = vectorizer.fit_transform(documents)
    
    # Apply K-means clustering
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42
    )
    labels = kmeans.fit_predict(vectors)
    
    return labels, kmeans.cluster_centers_

# Exercise 2: Hierarchical Document Clustering
def hierarchical_clustering(document_vectors: np.ndarray, 
                          max_display: int = 30) -> None:
    """
    Implement hierarchical clustering with dendrogram visualization
    
    Args:
        document_vectors: Document vectors to cluster
        max_display: Maximum number of samples to display
    """
    # Create linkage matrix
    linkage_matrix = linkage(document_vectors.toarray(), method='ward')
    
    # Create dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(
        linkage_matrix,
        truncate_mode='lastp',
        p=max_display,
        show_leaf_counts=True
    )
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index or (cluster size)')
    plt.ylabel('Distance')
    plt.show()

# Exercise 3: Cluster Evaluation
def evaluate_clustering(vectors: np.ndarray, 
                       labels: np.ndarray) -> Dict[str, float]:
    """
    Evaluate clustering quality using multiple metrics
    
    Args:
        vectors: Document vectors
        labels: Cluster labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Calculate evaluation metrics
    metrics = {
        'silhouette_score': silhouette_score(vectors, labels),
        'calinski_harabasz_score': calinski_harabasz_score(vectors, labels)
    }
    
    # Compute cluster distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        metrics[f'cluster_{label}_size'] = count
    
    return metrics

# Exercise 4: Topic Extraction
def extract_cluster_topics(documents: List[str], 
                         labels: np.ndarray,
                         vectorizer: TfidfVectorizer,
                         top_n: int = 5) -> Dict[int, List[str]]:
    """
    Extract main topics from document clusters
    
    Args:
        documents: List of documents
        labels: Cluster labels
        vectorizer: Fitted TF-IDF vectorizer
        top_n: Number of top terms to extract
        
    Returns:
        Dictionary mapping cluster labels to top terms
    """
    # Group documents by cluster
    cluster_docs = defaultdict(list)
    for doc, label in zip(documents, labels):
        cluster_docs[label].append(doc)
    
    # Extract top terms for each cluster
    feature_names = vectorizer.get_feature_names_out()
    cluster_topics = {}
    
    for label, docs in cluster_docs.items():
        # Get TF-IDF vectors for cluster documents
        cluster_vectors = vectorizer.transform(docs)
        
        # Calculate mean vector for cluster
        centroid = cluster_vectors.mean(axis=0).A1
        
        # Get top terms
        top_indices = centroid.argsort()[-top_n:][::-1]
        top_terms = [feature_names[i] for i in top_indices]
        cluster_topics[label] = top_terms
    
    return cluster_topics

# Exercise 5: Interactive Clustering Pipeline
class ClusteringPipeline:
    """Interactive document clustering pipeline"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.algorithms = {
            'kmeans': KMeans(n_clusters=5, random_state=42),
            'hierarchical': AgglomerativeClustering(n_clusters=5),
            'dbscan': DBSCAN(eps=0.5, min_samples=5)
        }
    
    def preprocess_documents(self, documents: List[str]) -> List[str]:
        """Preprocess documents"""
        processed = []
        for doc in documents:
            # Basic preprocessing
            doc = doc.lower()
            # Add more preprocessing steps as needed
            processed.append(doc)
        return processed
    
    def extract_features(self, processed_documents: List[str]) -> np.ndarray:
        """Extract document features"""
        return self.vectorizer.fit_transform(processed_documents)
    
    def apply_clustering(self, features: np.ndarray, 
                        algorithm: str = 'kmeans') -> np.ndarray:
        """Apply clustering algorithm"""
        if algorithm not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        return self.algorithms[algorithm].fit_predict(features)
    
    def evaluate_results(self, features: np.ndarray, 
                        labels: np.ndarray) -> Dict[str, float]:
        """Evaluate clustering results"""
        return evaluate_clustering(features, labels)
    
    def visualize_clusters(self, features: np.ndarray, 
                          labels: np.ndarray) -> None:
        """Visualize clustering results"""
        # Use t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        vectors_2d = tsne.fit_transform(features.toarray())
        
        # Plot clusters
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], 
                            c=labels, cmap='viridis')
        plt.colorbar(scatter)
        plt.title('Document Clusters')
        plt.show()
    
    def run(self, documents: List[str], 
            algorithm: str = 'kmeans') -> Dict[str, any]:
        """Run complete clustering pipeline"""
        # Process documents
        processed_docs = self.preprocess_documents(documents)
        
        # Extract features
        features = self.extract_features(processed_docs)
        
        # Apply clustering
        labels = self.apply_clustering(features, algorithm)
        
        # Evaluate results
        evaluation = self.evaluate_results(features, labels)
        
        # Extract topics
        topics = extract_cluster_topics(documents, labels, self.vectorizer)
        
        # Visualize results
        self.visualize_clusters(features, labels)
        
        return {
            'labels': labels,
            'evaluation': evaluation,
            'topics': topics
        }

# Exercise 7: Advanced Clustering
class EnsembleClusterer:
    """Ensemble clustering implementation"""
    
    def __init__(self, algorithms: List[BaseEstimator]):
        self.algorithms = algorithms
    
    def fit_predict(self, vectors: np.ndarray) -> np.ndarray:
        """
        Perform ensemble clustering
        
        Args:
            vectors: Document vectors
            
        Returns:
            Consensus cluster labels
        """
        # Get predictions from all algorithms
        predictions = []
        for algorithm in self.algorithms:
            pred = algorithm.fit_predict(vectors)
            predictions.append(pred)
        
        # Create consensus using majority voting
        consensus = np.zeros(len(vectors))
        for i in range(len(vectors)):
            votes = [pred[i] for pred in predictions]
            consensus[i] = max(set(votes), key=votes.count)
        
        return consensus
    
    def evaluate_consensus(self, predictions: List[np.ndarray]) -> float:
        """
        Evaluate consensus among different algorithms
        
        Args:
            predictions: List of cluster label arrays
            
        Returns:
            Mean agreement score
        """
        from sklearn.metrics import adjusted_rand_score
        scores = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                score = adjusted_rand_score(predictions[i], predictions[j])
                scores.append(score)
        return np.mean(scores)

# Exercise 8: Cluster-based Search
class ClusterBasedSearch:
    """Cluster-based document search implementation"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english'
        )
        self.clusterer = KMeans(n_clusters=5, random_state=42)
        self.documents = None
        self.vectors = None
        self.labels = None
    
    def index_documents(self, documents: List[str]) -> None:
        """
        Index documents for searching
        
        Args:
            documents: List of documents to index
        """
        self.documents = documents
        self.vectors = self.vectorizer.fit_transform(documents)
        self.labels = self.clusterer.fit_predict(self.vectors)
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (document, similarity score) tuples
        """
        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        # Find closest cluster
        cluster_centers = self.clusterer.cluster_centers_
        closest_cluster = np.argmin(
            [np.linalg.norm(query_vector.toarray() - center) 
             for center in cluster_centers]
        )
        
        # Get documents from closest cluster
        cluster_docs = np.where(self.labels == closest_cluster)[0]
        
        # Calculate similarities
        similarities = []
        for idx in cluster_docs:
            sim = np.dot(query_vector.toarray()[0], 
                        self.vectors[idx].toarray()[0])
            similarities.append((self.documents[idx], sim))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

# Example usage
if __name__ == "__main__":
    # Test data
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks for complex tasks",
        "Natural language processing deals with text analysis",
        "Neural networks are inspired by biological brains",
        "Text analysis is crucial for understanding documents",
        "Artificial intelligence is revolutionizing technology",
        "Document clustering helps organize information"
    ]
    
    # Test basic clustering
    labels, centers = cluster_documents(documents)
    print("Basic clustering labels:", labels)
    
    # Test pipeline
    pipeline = ClusteringPipeline()
    results = pipeline.run(documents)
    print("\nPipeline results:")
    print("Evaluation:", results['evaluation'])
    print("Topics:", results['topics'])
    
    # Test search
    searcher = ClusterBasedSearch()
    searcher.index_documents(documents)
    results = searcher.search("machine learning")
    print("\nSearch results for 'machine learning':")
    for doc, score in results:
        print(f"Score: {score:.3f} - {doc}") 