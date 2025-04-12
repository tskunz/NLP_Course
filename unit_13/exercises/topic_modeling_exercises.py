"""
Topic Modeling Exercises
This module contains exercises for practicing topic modeling concepts and implementations.
"""

from typing import List, Dict, Tuple
import numpy as np

def exercise_1_preprocessing(documents: List[str]) -> List[List[str]]:
    """
    Exercise 1: Document Preprocessing for Topic Modeling
    
    Implement preprocessing steps for topic modeling:
    1. Convert text to lowercase
    2. Remove punctuation and special characters
    3. Remove stopwords (use NLTK or spaCy stopwords)
    4. Perform lemmatization
    5. Remove short words (length < 3)
    
    Args:
        documents: List of document strings
        
    Returns:
        List of preprocessed and tokenized documents
    """
    # TODO: Implement preprocessing steps
    pass

def exercise_2_topic_coherence(topic_word_dist: List[List[Tuple[str, float]]],
                             documents: List[str]) -> float:
    """
    Exercise 2: Topic Coherence Calculation
    
    Implement a function to calculate topic coherence:
    1. For each topic's top N words
    2. Calculate word co-occurrence in documents
    3. Compute PMI-based coherence score
    
    Args:
        topic_word_dist: List of topics, each containing (word, probability) pairs
        documents: Original documents for co-occurrence calculation
        
    Returns:
        Coherence score
    """
    # TODO: Implement topic coherence calculation
    pass

def exercise_3_topic_similarity(topic1: List[Tuple[str, float]],
                              topic2: List[Tuple[str, float]]) -> float:
    """
    Exercise 3: Topic Similarity Measurement
    
    Implement a function to measure similarity between two topics:
    1. Convert topics to word-probability dictionaries
    2. Calculate cosine similarity or Jensen-Shannon divergence
    3. Return similarity score
    
    Args:
        topic1: First topic as list of (word, probability) pairs
        topic2: Second topic as list of (word, probability) pairs
        
    Returns:
        Similarity score between topics
    """
    # TODO: Implement topic similarity measurement
    pass

def exercise_4_dynamic_topics(time_sliced_docs: List[List[str]],
                            num_topics: int = 5) -> List[List[List[Tuple[str, float]]]]:
    """
    Exercise 4: Dynamic Topic Modeling
    
    Implement dynamic topic modeling to track topic evolution:
    1. Process documents in time slices
    2. Track topic changes across time periods
    3. Identify emerging and dying topics
    
    Args:
        time_sliced_docs: List of document lists for each time period
        num_topics: Number of topics to extract
        
    Returns:
        List of topic distributions for each time period
    """
    # TODO: Implement dynamic topic modeling
    pass

def exercise_5_topic_labeling(topic_words: List[List[Tuple[str, float]]]) -> List[str]:
    """
    Exercise 5: Automatic Topic Labeling
    
    Implement automatic topic labeling:
    1. Analyze word distributions in topics
    2. Use word embeddings or knowledge bases
    3. Generate descriptive labels
    
    Args:
        topic_words: List of topics with their word distributions
        
    Returns:
        List of topic labels
    """
    # TODO: Implement automatic topic labeling
    pass

def run_tests():
    """Run tests for the exercises."""
    # Test data
    documents = [
        "Machine learning algorithms are transforming artificial intelligence research.",
        "Deep neural networks achieve state-of-the-art results in computer vision tasks.",
        "Natural language processing helps computers understand human language.",
        "Reinforcement learning enables agents to learn from their environment.",
        "Data science combines statistics and programming to analyze big data."
    ]
    
    # Test Exercise 1
    print("\nTesting Exercise 1: Document Preprocessing")
    processed_docs = exercise_1_preprocessing(documents)
    if processed_docs:
        print("Processed documents:")
        for doc in processed_docs[:2]:
            print(f"  {doc}")
    
    # Test Exercise 2
    print("\nTesting Exercise 2: Topic Coherence")
    sample_topics = [
        [("learning", 0.1), ("machine", 0.08), ("algorithm", 0.06)],
        [("data", 0.1), ("analysis", 0.07), ("science", 0.05)]
    ]
    coherence = exercise_2_topic_coherence(sample_topics, documents)
    if coherence:
        print(f"Topic coherence score: {coherence:.4f}")
    
    # Test Exercise 3
    print("\nTesting Exercise 3: Topic Similarity")
    topic1 = [("machine", 0.1), ("learning", 0.08), ("algorithm", 0.06)]
    topic2 = [("deep", 0.1), ("learning", 0.08), ("neural", 0.06)]
    similarity = exercise_3_topic_similarity(topic1, topic2)
    if similarity:
        print(f"Topic similarity score: {similarity:.4f}")
    
    # Test Exercise 4
    print("\nTesting Exercise 4: Dynamic Topics")
    time_slices = [
        documents[:2],
        documents[2:4],
        documents[4:]
    ]
    dynamic_topics = exercise_4_dynamic_topics(time_slices)
    if dynamic_topics:
        print("Dynamic topic evolution:")
        for t, topics in enumerate(dynamic_topics):
            print(f"\nTime period {t + 1}:")
            for idx, topic in enumerate(topics):
                print(f"  Topic {idx + 1}: {[word for word, _ in topic[:3]]}")
    
    # Test Exercise 5
    print("\nTesting Exercise 5: Topic Labeling")
    sample_topics = [
        [("machine", 0.1), ("learning", 0.08), ("algorithm", 0.06)],
        [("data", 0.1), ("analysis", 0.07), ("science", 0.05)],
        [("neural", 0.1), ("network", 0.08), ("deep", 0.06)]
    ]
    labels = exercise_5_topic_labeling(sample_topics)
    if labels:
        print("Generated topic labels:")
        for idx, label in enumerate(labels):
            print(f"  Topic {idx + 1}: {label}")

if __name__ == "__main__":
    run_tests() 