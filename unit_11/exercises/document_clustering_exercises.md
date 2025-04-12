# Document Clustering Exercises

## Exercise 1: Basic Document Clustering
Implement a basic document clustering solution using the K-means algorithm.

### Task
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def cluster_documents(documents, n_clusters=5):
    """
    TODO: Implement document clustering using TF-IDF and K-means
    1. Create TF-IDF vectors for the documents
    2. Apply K-means clustering
    3. Return document labels and cluster centers
    """
    pass

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

# Expected usage
labels = cluster_documents(documents)
```

## Exercise 2: Hierarchical Document Clustering
Implement hierarchical clustering and create a dendrogram visualization.

### Task
```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

def hierarchical_clustering(document_vectors):
    """
    TODO: Implement hierarchical clustering
    1. Create linkage matrix
    2. Generate dendrogram
    3. Plot and save visualization
    """
    pass

# Test with TF-IDF vectors from Exercise 1
```

## Exercise 3: Cluster Evaluation
Implement functions to evaluate clustering quality using different metrics.

### Task
```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def evaluate_clustering(vectors, labels):
    """
    TODO: Implement clustering evaluation
    1. Calculate silhouette score
    2. Calculate Calinski-Harabasz score
    3. Compute cluster sizes and distribution
    4. Return evaluation metrics
    """
    pass

# Test with results from Exercises 1 and 2
```

## Exercise 4: Topic Extraction
Extract and analyze main topics from document clusters.

### Task
```python
def extract_cluster_topics(documents, labels, vectorizer, top_n=5):
    """
    TODO: Implement topic extraction
    1. Group documents by cluster
    2. Extract top terms for each cluster
    3. Generate topic summaries
    4. Return cluster topics
    """
    pass

# Example usage
topics = extract_cluster_topics(documents, labels, vectorizer)
```

## Exercise 5: Interactive Clustering Pipeline
Create an interactive pipeline that combines vectorization, clustering, and evaluation.

### Task
```python
class ClusteringPipeline:
    """
    TODO: Implement an interactive clustering pipeline
    1. Document preprocessing
    2. Feature extraction
    3. Clustering with multiple algorithms
    4. Evaluation and visualization
    5. Result comparison
    """
    def __init__(self):
        pass
    
    def preprocess_documents(self, documents):
        pass
    
    def extract_features(self, processed_documents):
        pass
    
    def apply_clustering(self, features, algorithm='kmeans'):
        pass
    
    def evaluate_results(self, features, labels):
        pass
    
    def visualize_clusters(self, features, labels):
        pass

# Example usage
pipeline = ClusteringPipeline()
results = pipeline.run(documents)
```

## Exercise 6: Real-world Application
Apply document clustering to a real dataset (e.g., news articles, scientific papers).

### Task
1. Download or prepare a dataset of documents
2. Implement a complete clustering solution:
   - Document preprocessing
   - Feature extraction
   - Clustering
   - Evaluation
   - Visualization
3. Analyze and interpret the results
4. Write a report on findings

## Exercise 7: Advanced Clustering
Implement an ensemble clustering approach combining multiple algorithms.

### Task
```python
class EnsembleClusterer:
    """
    TODO: Implement ensemble clustering
    1. Combine multiple clustering algorithms
    2. Implement consensus mechanism
    3. Handle cluster label alignment
    4. Evaluate ensemble performance
    """
    def __init__(self, algorithms):
        pass
    
    def fit_predict(self, vectors):
        pass
    
    def evaluate_consensus(self, predictions):
        pass

# Test with previous exercises
```

## Exercise 8: Cluster-based Search
Implement a search function that uses document clusters to improve search results.

### Task
```python
class ClusterBasedSearch:
    """
    TODO: Implement cluster-based search
    1. Index documents using clusters
    2. Process search queries
    3. Rank results using cluster information
    4. Return relevant documents
    """
    def __init__(self):
        pass
    
    def index_documents(self, documents):
        pass
    
    def search(self, query, top_k=5):
        pass

# Example usage
searcher = ClusterBasedSearch()
searcher.index_documents(documents)
results = searcher.search("machine learning")
```

## Solutions

The solutions to these exercises can be found in the `solutions` directory. However, we strongly encourage you to attempt the exercises before checking the solutions.

### Evaluation Criteria
Your solutions will be evaluated based on:
1. Correctness of implementation
2. Code quality and organization
3. Documentation and comments
4. Performance and efficiency
5. Error handling
6. Test coverage

## Additional Resources
1. Scikit-learn documentation: [Clustering](https://scikit-learn.org/stable/modules/clustering.html)
2. Python Data Science Handbook: [Clustering](https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html)
3. Research papers on document clustering techniques

## Submission Guidelines
1. Create a new Python file for each exercise
2. Include docstrings and comments
3. Add unit tests for your implementations
4. Provide a brief report on your approach and results
5. Submit all files in a zip archive

---
*Note: These exercises are designed to build practical skills in document clustering. Take time to understand each concept before moving to the next exercise.* 