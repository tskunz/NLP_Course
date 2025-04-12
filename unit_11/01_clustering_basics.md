# Introduction to Document Clustering

## Overview
Document clustering is a technique in Natural Language Processing that groups similar documents together based on their content. This unsupervised learning approach helps organize and analyze large collections of text documents.

## Basic Concepts

### 1. Document Representation
```python
from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

class DocumentVectorizer:
    def __init__(self, max_features: int = 1000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def vectorize_documents(self, documents: List[str]) -> np.ndarray:
        """Convert documents to TF-IDF vectors"""
        vectors = self.vectorizer.fit_transform(documents)
        return normalize(vectors, norm='l2')
    
    def get_features(self) -> List[str]:
        """Get feature names (terms) used in vectorization"""
        return self.vectorizer.get_feature_names_out()
    
    def get_document_terms(self, doc_vector: np.ndarray) -> List[Tuple[str, float]]:
        """Get most important terms for a document"""
        features = self.get_features()
        term_scores = [(features[i], score) 
                      for i, score in enumerate(doc_vector.toarray()[0])
                      if score > 0]
        return sorted(term_scores, key=lambda x: x[1], reverse=True)
```

### 2. Distance Metrics
```python
class DistanceMetrics:
    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    @staticmethod
    def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate Euclidean distance between vectors"""
        return np.sqrt(np.sum((v1 - v2) ** 2))
    
    @staticmethod
    def manhattan_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate Manhattan distance between vectors"""
        return np.sum(np.abs(v1 - v2))
    
    @staticmethod
    def jaccard_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate Jaccard similarity between vectors"""
        intersection = np.sum(np.minimum(v1, v2))
        union = np.sum(np.maximum(v1, v2))
        return intersection / union if union > 0 else 0.0
```

### 3. Feature Representation
```python
class FeatureExtractor:
    def __init__(self):
        self.vectorizer = DocumentVectorizer()
        
    def extract_features(self, documents: List[str]) -> Dict[str, np.ndarray]:
        """Extract multiple feature representations"""
        # TF-IDF vectors
        tfidf_vectors = self.vectorizer.vectorize_documents(documents)
        
        # Document statistics
        doc_stats = []
        for doc in documents:
            words = doc.split()
            unique_words = set(words)
            stats = np.array([
                len(words),                    # Document length
                len(unique_words),             # Vocabulary size
                len(words) / len(unique_words) # Type-token ratio
            ])
            doc_stats.append(stats)
        
        return {
            'tfidf': tfidf_vectors,
            'stats': np.array(doc_stats)
        }
```

## Feature Selection

### 1. Term Selection
```python
class TermSelector:
    def __init__(self, min_df: float = 0.01, max_df: float = 0.95):
        self.min_df = min_df
        self.max_df = max_df
        
    def select_terms(self, term_doc_matrix: np.ndarray, 
                    terms: List[str]) -> List[str]:
        """Select relevant terms based on document frequency"""
        doc_freq = np.sum(term_doc_matrix > 0, axis=0) / term_doc_matrix.shape[0]
        selected_indices = np.where(
            (doc_freq >= self.min_df) & (doc_freq <= self.max_df)
        )[0]
        return [terms[i] for i in selected_indices]
```

### 2. Dimensionality Reduction
```python
from sklearn.decomposition import TruncatedSVD

class DimensionalityReducer:
    def __init__(self, n_components: int = 100):
        self.svd = TruncatedSVD(n_components=n_components)
        
    def reduce_dimensions(self, vectors: np.ndarray) -> np.ndarray:
        """Reduce dimensionality using SVD"""
        return self.svd.fit_transform(vectors)
    
    def get_explained_variance(self) -> float:
        """Get explained variance ratio"""
        return np.sum(self.svd.explained_variance_ratio_)
```

## Evaluation Metrics

### 1. Internal Metrics
```python
class ClusterEvaluator:
    @staticmethod
    def silhouette_score(vectors: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering"""
        from sklearn.metrics import silhouette_score
        return silhouette_score(vectors, labels)
    
    @staticmethod
    def calinski_harabasz_score(vectors: np.ndarray, 
                               labels: np.ndarray) -> float:
        """Calculate Calinski-Harabasz score"""
        from sklearn.metrics import calinski_harabasz_score
        return calinski_harabasz_score(vectors, labels)
    
    @staticmethod
    def davies_bouldin_score(vectors: np.ndarray, 
                            labels: np.ndarray) -> float:
        """Calculate Davies-Bouldin score"""
        from sklearn.metrics import davies_bouldin_score
        return davies_bouldin_score(vectors, labels)
```

### 2. External Metrics
```python
class ExternalEvaluator:
    @staticmethod
    def adjusted_rand_score(true_labels: np.ndarray, 
                          pred_labels: np.ndarray) -> float:
        """Calculate Adjusted Rand Index"""
        from sklearn.metrics import adjusted_rand_score
        return adjusted_rand_score(true_labels, pred_labels)
    
    @staticmethod
    def normalized_mutual_info(true_labels: np.ndarray, 
                             pred_labels: np.ndarray) -> float:
        """Calculate Normalized Mutual Information"""
        from sklearn.metrics import normalized_mutual_info_score
        return normalized_mutual_info_score(true_labels, pred_labels)
```

## Best Practices

### 1. Data Preparation
- Text preprocessing
- Feature selection
- Normalization

### 2. Algorithm Selection
- Data characteristics
- Scalability requirements
- Interpretability needs

### 3. Evaluation Strategy
- Multiple metrics
- Cross-validation
- Error analysis

## Applications

### 1. Document Organization
- Topic discovery
- Content categorization
- Archive management

### 2. Information Retrieval
- Search result clustering
- Document recommendation
- Content filtering

### 3. Text Analysis
- Trend analysis
- Pattern discovery
- Anomaly detection

## References
1. Manning, C. et al. "Introduction to Information Retrieval"
2. Aggarwal, C.C. & Zhai, C. "Mining Text Data"
3. Xu, R. & Wunsch, D. "Survey of Clustering Algorithms"

---
*Note: This document introduces document clustering concepts with practical Python implementations. The code examples are simplified for illustration purposes.* 