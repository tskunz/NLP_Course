"""
Solutions for Semantic Similarity Exercises
This module contains solutions for the semantic similarity exercises.
"""

from typing import List, Dict, Tuple, Set
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from collections import defaultdict

def exercise_1_wordnet_similarity(word1: str, word2: str) -> Dict[str, float]:
    """Solution for Exercise 1: WordNet-based Similarity"""
    # Get synsets for both words
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)
    
    # Initialize scores
    path_max = 0.0
    lch_max = 0.0
    wup_max = 0.0
    
    # If either word has no synsets, return zeros
    if not synsets1 or not synsets2:
        return {
            'path_similarity': 0.0,
            'lch_similarity': 0.0,
            'wup_similarity': 0.0
        }
    
    # Compare each pair of synsets
    for s1 in synsets1:
        for s2 in synsets2:
            # Path similarity
            path_sim = s1.path_similarity(s2)
            if path_sim and path_sim > path_max:
                path_max = path_sim
            
            # Leacock-Chodorow similarity
            try:
                lch_sim = s1.lch_similarity(s2)
                if lch_sim and lch_sim > lch_max:
                    lch_max = lch_sim
            except:
                pass
            
            # Wu-Palmer similarity
            try:
                wup_sim = s1.wup_similarity(s2)
                if wup_sim and wup_sim > wup_max:
                    wup_max = wup_sim
            except:
                pass
    
    return {
        'path_similarity': path_max,
        'lch_similarity': lch_max,
        'wup_similarity': wup_max
    }

def exercise_2_word_embeddings(sentences: List[str]) -> Dict[str, float]:
    """Solution for Exercise 2: Word Embeddings"""
    # Tokenize sentences
    tokenized_sentences = [word_tokenize(sent.lower()) for sent in sentences]
    
    # Train Word2Vec model
    model = Word2Vec(sentences=tokenized_sentences,
                    vector_size=100,
                    window=5,
                    min_count=1,
                    workers=4)
    
    # Compute word similarities
    similarity_scores = []
    analogy_scores = []
    
    # Test word similarities
    test_pairs = [
        ('python', 'programming'),
        ('machine', 'learning'),
        ('quick', 'fast')
    ]
    
    for word1, word2 in test_pairs:
        try:
            sim = model.wv.similarity(word1, word2)
            similarity_scores.append(sim)
        except KeyError:
            continue
    
    # Test word analogies
    test_analogies = [
        ('king', 'queen', 'man', 'woman'),
        ('python', 'programming', 'java', 'coding')
    ]
    
    for w1, w2, w3, w4 in test_analogies:
        try:
            # Compute analogy score
            result = model.wv.most_similar(
                positive=[w2, w3],
                negative=[w1]
            )
            
            # Check if expected word is in top results
            for word, score in result:
                if word == w4:
                    analogy_scores.append(score)
                    break
        except KeyError:
            continue
    
    return {
        'similarity_score': np.mean(similarity_scores) if similarity_scores else 0.0,
        'analogy_score': np.mean(analogy_scores) if analogy_scores else 0.0
    }

def exercise_3_document_vectors(documents: List[str]) -> np.ndarray:
    """Solution for Exercise 3: Document Vector Representations"""
    # Create TF-IDF vectors
    tfidf = TfidfVectorizer()
    tfidf_vectors = tfidf.fit_transform(documents)
    
    # Create word embeddings
    tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
    w2v_model = Word2Vec(sentences=tokenized_docs,
                        vector_size=100,
                        window=5,
                        min_count=1,
                        workers=4)
    
    # Create document vectors using weighted word embeddings
    doc_vectors = []
    for doc in documents:
        words = word_tokenize(doc.lower())
        word_vectors = []
        
        for word in words:
            try:
                # Get word vector and weight it by TF-IDF if available
                vector = w2v_model.wv[word]
                tfidf_weight = tfidf.vocabulary_.get(word, 1.0)
                word_vectors.append(vector * tfidf_weight)
            except KeyError:
                continue
        
        if word_vectors:
            # Average the word vectors
            doc_vector = np.mean(word_vectors, axis=0)
        else:
            doc_vector = np.zeros(100)
        
        doc_vectors.append(doc_vector)
    
    # Convert to numpy array
    doc_vectors = np.array(doc_vectors)
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(doc_vectors)
    
    return similarity_matrix

def exercise_4_semantic_search(query: str, documents: List[str]) -> List[Tuple[float, str]]:
    """Solution for Exercise 4: Semantic Search"""
    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer()
    
    # Fit on documents and transform both query and documents
    all_texts = [query] + documents
    tfidf_vectors = tfidf.fit_transform(all_texts)
    
    # Split into query and document vectors
    query_vector = tfidf_vectors[0:1]
    doc_vectors = tfidf_vectors[1:]
    
    # Create word embeddings
    tokenized_texts = [word_tokenize(text.lower()) for text in all_texts]
    w2v_model = Word2Vec(sentences=tokenized_texts,
                        vector_size=100,
                        window=5,
                        min_count=1,
                        workers=4)
    
    # Create document vectors using word embeddings
    doc_vectors_w2v = []
    query_words = word_tokenize(query.lower())
    query_vector_w2v = np.mean([w2v_model.wv[w] for w in query_words 
                               if w in w2v_model.wv], axis=0)
    
    for doc in documents:
        words = word_tokenize(doc.lower())
        vectors = [w2v_model.wv[w] for w in words if w in w2v_model.wv]
        if vectors:
            doc_vectors_w2v.append(np.mean(vectors, axis=0))
        else:
            doc_vectors_w2v.append(np.zeros(100))
    
    # Compute similarities using both methods
    tfidf_similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    w2v_similarities = cosine_similarity([query_vector_w2v], doc_vectors_w2v).flatten()
    
    # Combine similarities (simple average)
    combined_similarities = (tfidf_similarities + w2v_similarities) / 2
    
    # Create and sort results
    results = list(zip(combined_similarities, documents))
    return sorted(results, reverse=True)

def exercise_5_similarity_evaluation(predictions: List[float], 
                                  gold_standard: List[float]) -> Dict[str, float]:
    """Solution for Exercise 5: Similarity Evaluation"""
    # Convert inputs to numpy arrays
    pred = np.array(predictions)
    gold = np.array(gold_standard)
    
    # Compute Pearson correlation
    pearson_corr, _ = pearsonr(pred, gold)
    
    # Compute Spearman correlation
    spearman_corr, _ = spearmanr(pred, gold)
    
    # Compute Mean Squared Error
    mse = np.mean((pred - gold) ** 2)
    
    # Handle NaN values
    metrics = {
        'pearson': float(pearson_corr) if not np.isnan(pearson_corr) else 0.0,
        'spearman': float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
        'mse': float(mse)
    }
    
    return metrics

def main():
    """Run examples of the solutions"""
    # Example 1: WordNet Similarity
    print("\n1. WordNet Similarity Example:")
    word_pairs = [
        ('car', 'vehicle'),
        ('happy', 'sad'),
        ('python', 'snake'),
        ('computer', 'machine')
    ]
    
    for w1, w2 in word_pairs:
        scores = exercise_1_wordnet_similarity(w1, w2)
        print(f"\n{w1} - {w2}:")
        for measure, score in scores.items():
            print(f"  {measure}: {score:.3f}")
    
    # Example 2: Word Embeddings
    print("\n2. Word Embeddings Example:")
    sentences = [
        "the quick brown fox jumps over the lazy dog",
        "python is a popular programming language",
        "machine learning algorithms are powerful",
        "natural language processing is fascinating"
    ]
    scores = exercise_2_word_embeddings(sentences)
    print("\nScores:")
    for metric, score in scores.items():
        print(f"  {metric}: {score:.3f}")
    
    # Example 3: Document Vectors
    print("\n3. Document Vectors Example:")
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "Python is a popular programming language",
        "Natural language processing is fascinating",
        "Machine learning algorithms are powerful"
    ]
    similarity_matrix = exercise_3_document_vectors(documents)
    print("\nDocument Similarity Matrix:")
    print(similarity_matrix)
    
    # Example 4: Semantic Search
    print("\n4. Semantic Search Example:")
    query = "programming languages and algorithms"
    results = exercise_4_semantic_search(query, documents)
    print("\nSearch Results:")
    for score, doc in results:
        print(f"\n{score:.3f}: {doc}")
    
    # Example 5: Similarity Evaluation
    print("\n5. Similarity Evaluation Example:")
    predictions = [0.5, 0.7, 0.3, 0.8, 0.9]
    gold_standard = [0.6, 0.8, 0.2, 0.8, 0.9]
    metrics = exercise_5_similarity_evaluation(predictions, gold_standard)
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")

if __name__ == "__main__":
    # Download required NLTK data
    nltk.download('wordnet')
    nltk.download('punkt')
    
    # Run the examples
    main() 