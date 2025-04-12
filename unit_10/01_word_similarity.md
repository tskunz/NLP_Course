# Word Similarity Measures

## Overview
Word similarity is a fundamental concept in Natural Language Processing that quantifies how similar or related two words are. This can be measured through various approaches, including lexical databases, distributional semantics, and neural embeddings.

## Similarity Metrics

### 1. String-Based Similarity
```python
from typing import List, Dict, Tuple
import numpy as np
from difflib import SequenceMatcher

class StringSimilarity:
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return StringSimilarity.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @staticmethod
    def sequence_similarity(s1: str, s2: str) -> float:
        """Calculate sequence similarity ratio"""
        return SequenceMatcher(None, s1, s2).ratio()
```

### 2. WordNet-Based Similarity
```python
from nltk.corpus import wordnet as wn

class WordNetSimilarity:
    @staticmethod
    def path_similarity(word1: str, word2: str) -> float:
        """Calculate path similarity using WordNet"""
        synsets1 = wn.synsets(word1)
        synsets2 = wn.synsets(word2)
        
        if not synsets1 or not synsets2:
            return 0.0
        
        max_sim = 0.0
        for syn1 in synsets1:
            for syn2 in synsets2:
                sim = syn1.path_similarity(syn2)
                if sim and sim > max_sim:
                    max_sim = sim
        
        return max_sim
    
    @staticmethod
    def wup_similarity(word1: str, word2: str) -> float:
        """Calculate Wu-Palmer similarity"""
        synsets1 = wn.synsets(word1)
        synsets2 = wn.synsets(word2)
        
        if not synsets1 or not synsets2:
            return 0.0
        
        max_sim = 0.0
        for syn1 in synsets1:
            for syn2 in synsets2:
                sim = syn1.wup_similarity(syn2)
                if sim and sim > max_sim:
                    max_sim = sim
        
        return max_sim
```

### 3. Vector-Based Similarity
```python
class VectorSimilarity:
    def __init__(self, model_path: str):
        self.word_vectors = self.load_vectors(model_path)
    
    def load_vectors(self, path: str) -> Dict[str, np.ndarray]:
        """Load word vectors from file"""
        vectors = {}
        with open(path, 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                vectors[word] = vector
        return vectors
    
    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return dot_product / (norm_v1 * norm_v2)
    
    def word_similarity(self, word1: str, word2: str) -> float:
        """Calculate similarity between two words"""
        if word1 not in self.word_vectors or word2 not in self.word_vectors:
            return 0.0
        
        v1 = self.word_vectors[word1]
        v2 = self.word_vectors[word2]
        return self.cosine_similarity(v1, v2)
```

## Distance Measures

### 1. Euclidean Distance
```python
def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate Euclidean distance between vectors"""
    return np.sqrt(np.sum((v1 - v2) ** 2))
```

### 2. Manhattan Distance
```python
def manhattan_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate Manhattan distance between vectors"""
    return np.sum(np.abs(v1 - v2))
```

## Evaluation Methods

### 1. Correlation Analysis
```python
class SimilarityEvaluator:
    def __init__(self):
        self.human_scores = {}  # Reference similarity scores
        
    def load_dataset(self, path: str) -> None:
        """Load similarity dataset with human judgments"""
        with open(path, 'r') as f:
            for line in f:
                word1, word2, score = line.strip().split('\t')
                self.human_scores[(word1, word2)] = float(score)
    
    def evaluate_correlation(self, similarity_func) -> Dict[str, float]:
        """Calculate correlation with human judgments"""
        system_scores = []
        human_scores = []
        
        for (word1, word2), score in self.human_scores.items():
            sys_score = similarity_func(word1, word2)
            system_scores.append(sys_score)
            human_scores.append(score)
        
        return {
            'pearson': np.corrcoef(system_scores, human_scores)[0, 1],
            'spearman': spearmanr(system_scores, human_scores)[0]
        }
```

## Best Practices

### 1. Preprocessing
- Word normalization
- Case handling
- Out-of-vocabulary words

### 2. Model Selection
- Task requirements
- Language specifics
- Performance needs

### 3. Evaluation
- Multiple metrics
- Task-specific evaluation
- Error analysis

## Applications

### 1. Information Retrieval
- Query expansion
- Document similarity
- Semantic search

### 2. Text Classification
- Feature engineering
- Document clustering
- Topic modeling

### 3. Machine Translation
- Word alignment
- Phrase similarity
- Cross-lingual mapping

## References
1. Miller, G.A. "WordNet: A Lexical Database for English"
2. Mikolov, T. et al. "Efficient Estimation of Word Representations in Vector Space"
3. Pennington, J. et al. "GloVe: Global Vectors for Word Representation"

---
*Note: This document introduces word similarity measures with practical Python implementations. The code examples are simplified for illustration purposes.* 