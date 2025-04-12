"""
Semantic Similarity Examples
This module demonstrates various approaches to computing semantic similarity
between words and documents.
"""

from typing import List, Dict, Tuple, Union
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from collections import defaultdict

class WordSimilarity:
    """Implements various word similarity measures"""
    
    def __init__(self):
        """Initialize word similarity models"""
        # Download required NLTK data
        nltk.download('wordnet')
        nltk.download('punkt')
        
        # Initialize sentence transformer
        self.transformer = SentenceTransformer('all-MiniLM-L6-v2')
    
    def path_similarity(self, word1: str, word2: str) -> float:
        """
        Compute path-based similarity using WordNet
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Similarity score between 0 and 1
        """
        # Get all synsets for both words
        synsets1 = wn.synsets(word1)
        synsets2 = wn.synsets(word2)
        
        if not synsets1 or not synsets2:
            return 0.0
        
        # Find maximum similarity between any pair of synsets
        max_sim = 0.0
        for s1 in synsets1:
            for s2 in synsets2:
                sim = s1.path_similarity(s2)
                if sim and sim > max_sim:
                    max_sim = sim
        
        return max_sim
    
    def train_word2vec(self, sentences: List[List[str]], 
                      vector_size: int = 100) -> Word2Vec:
        """
        Train a Word2Vec model on the given sentences
        
        Args:
            sentences: List of tokenized sentences
            vector_size: Dimension of word vectors
            
        Returns:
            Trained Word2Vec model
        """
        model = Word2Vec(sentences=sentences,
                        vector_size=vector_size,
                        window=5,
                        min_count=1,
                        workers=4)
        return model
    
    def compute_word_similarity(self, word1: str, word2: str, 
                              method: str = 'transformer') -> float:
        """
        Compute word similarity using different methods
        
        Args:
            word1: First word
            word2: Second word
            method: Similarity method ('transformer', 'wordnet')
            
        Returns:
            Similarity score between 0 and 1
        """
        if method == 'transformer':
            # Use sentence transformer
            embeddings = self.transformer.encode([word1, word2])
            return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
        
        elif method == 'wordnet':
            return self.path_similarity(word1, word2)
        
        else:
            raise ValueError(f"Unknown method: {method}")

class DocumentSimilarity:
    """Implements document similarity measures"""
    
    def __init__(self):
        """Initialize document similarity models"""
        self.vectorizer = TfidfVectorizer()
        self.transformer = SentenceTransformer('all-MiniLM-L6-v2')
    
    def compute_tfidf_similarity(self, doc1: str, doc2: str) -> float:
        """
        Compute TF-IDF based cosine similarity between documents
        
        Args:
            doc1: First document
            doc2: Second document
            
        Returns:
            Similarity score between 0 and 1
        """
        # Create TF-IDF vectors
        vectors = self.vectorizer.fit_transform([doc1, doc2])
        
        # Compute cosine similarity
        return float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0])
    
    def compute_semantic_similarity(self, doc1: str, doc2: str) -> float:
        """
        Compute semantic similarity using sentence transformers
        
        Args:
            doc1: First document
            doc2: Second document
            
        Returns:
            Similarity score between 0 and 1
        """
        # Create document embeddings
        embeddings = self.transformer.encode([doc1, doc2])
        
        # Compute cosine similarity
        return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
    
    def find_similar_documents(self, query: str, documents: List[str], 
                             method: str = 'semantic') -> List[Tuple[float, str]]:
        """
        Find documents similar to a query
        
        Args:
            query: Query document
            documents: List of documents to search
            method: Similarity method ('tfidf', 'semantic')
            
        Returns:
            List of (similarity_score, document) pairs, sorted by similarity
        """
        if method == 'tfidf':
            # Create TF-IDF vectors
            vectors = self.vectorizer.fit_transform([query] + documents)
            query_vector = vectors[0:1]
            doc_vectors = vectors[1:]
            
            # Compute similarities
            similarities = cosine_similarity(query_vector, doc_vectors)[0]
            
        elif method == 'semantic':
            # Create embeddings
            embeddings = self.transformer.encode([query] + documents)
            query_embedding = embeddings[0]
            doc_embeddings = embeddings[1:]
            
            # Compute similarities
            similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Sort by similarity
        return sorted(zip(similarities, documents), reverse=True)

class Doc2VecSimilarity:
    """Implements Doc2Vec-based document similarity"""
    
    def __init__(self, vector_size: int = 100):
        """
        Initialize Doc2Vec model
        
        Args:
            vector_size: Dimension of document vectors
        """
        self.vector_size = vector_size
        self.model = None
    
    def train(self, documents: List[str]):
        """
        Train Doc2Vec model on documents
        
        Args:
            documents: List of documents to train on
        """
        # Prepare tagged documents
        tagged_docs = [
            TaggedDocument(words=word_tokenize(doc.lower()), 
                         tags=[str(i)]) 
            for i, doc in enumerate(documents)
        ]
        
        # Train model
        self.model = Doc2Vec(documents=tagged_docs,
                           vector_size=self.vector_size,
                           window=5,
                           min_count=1,
                           workers=4)
    
    def infer_vector(self, document: str) -> np.ndarray:
        """
        Infer vector for a new document
        
        Args:
            document: Document to infer vector for
            
        Returns:
            Document vector
        """
        if not self.model:
            raise ValueError("Model not trained")
        
        return self.model.infer_vector(word_tokenize(document.lower()))
    
    def find_similar_documents(self, query: str, documents: List[str], 
                             topn: int = 5) -> List[Tuple[float, str]]:
        """
        Find documents similar to a query
        
        Args:
            query: Query document
            documents: List of documents to search
            topn: Number of similar documents to return
            
        Returns:
            List of (similarity_score, document) pairs, sorted by similarity
        """
        if not self.model:
            raise ValueError("Model not trained")
        
        # Infer vector for query
        query_vec = self.infer_vector(query)
        
        # Compute similarities with all documents
        similarities = []
        for i, doc in enumerate(documents):
            doc_vec = self.model.dv[str(i)]
            sim = float(cosine_similarity([query_vec], [doc_vec])[0][0])
            similarities.append((sim, doc))
        
        return sorted(similarities, reverse=True)[:topn]

def main():
    """Demonstrate usage of similarity measures"""
    # Example texts
    words = ['computer', 'laptop', 'keyboard', 'tree']
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "A fast brown fox leaps across a sleepy canine",
        "Python is a popular programming language",
        "Java is another widely used programming language",
        "Natural language processing is fascinating"
    ]
    
    # Word similarity examples
    print("\n1. Word Similarity Examples:")
    word_sim = WordSimilarity()
    
    print("\nWordNet path similarity:")
    for w1 in words[:2]:
        for w2 in words:
            sim = word_sim.path_similarity(w1, w2)
            print(f"{w1} - {w2}: {sim:.3f}")
    
    print("\nTransformer-based word similarity:")
    for w1 in words[:2]:
        for w2 in words:
            sim = word_sim.compute_word_similarity(w1, w2, method='transformer')
            print(f"{w1} - {w2}: {sim:.3f}")
    
    # Document similarity examples
    print("\n2. Document Similarity Examples:")
    doc_sim = DocumentSimilarity()
    
    print("\nTF-IDF similarity:")
    sim = doc_sim.compute_tfidf_similarity(documents[0], documents[1])
    print(f"Similarity between first two documents: {sim:.3f}")
    
    print("\nSemantic similarity:")
    sim = doc_sim.compute_semantic_similarity(documents[0], documents[1])
    print(f"Similarity between first two documents: {sim:.3f}")
    
    print("\nSimilar documents to query:")
    query = "Programming languages are essential tools"
    similar_docs = doc_sim.find_similar_documents(query, documents)
    for sim, doc in similar_docs:
        print(f"{sim:.3f}: {doc}")
    
    # Doc2Vec examples
    print("\n3. Doc2Vec Examples:")
    d2v = Doc2VecSimilarity(vector_size=50)
    d2v.train(documents)
    
    print("\nSimilar documents using Doc2Vec:")
    similar_docs = d2v.find_similar_documents(query, documents)
    for sim, doc in similar_docs:
        print(f"{sim:.3f}: {doc}")

if __name__ == "__main__":
    main() 