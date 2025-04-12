"""
Topic Modeling Exercise Solutions
This module contains solutions for the topic modeling exercises.
"""

from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict, Counter
import spacy
import re
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def exercise_1_preprocessing(documents: List[str]) -> List[List[str]]:
    """Solution for Exercise 1: Document Preprocessing"""
    # Load spaCy model
    nlp = spacy.load('en_core_web_sm')
    
    processed_docs = []
    for doc in documents:
        # Process with spaCy
        doc = nlp(doc.lower())
        
        # Apply preprocessing steps
        tokens = [
            token.lemma_  # Lemmatization
            for token in doc
            if (
                not token.is_stop and  # Remove stopwords
                not token.is_punct and  # Remove punctuation
                not token.is_space and  # Remove whitespace
                len(token.text) > 2 and  # Remove short words
                not token.like_num and  # Remove numbers
                not bool(re.match(r'^[^a-zA-Z]+$', token.text))  # Remove special characters
            )
        ]
        
        processed_docs.append(tokens)
    
    return processed_docs

def exercise_2_topic_coherence(topic_word_dist: List[List[Tuple[str, float]]],
                             documents: List[str]) -> float:
    """Solution for Exercise 2: Topic Coherence Calculation"""
    # Preprocess documents
    processed_docs = exercise_1_preprocessing(documents)
    
    # Create word co-occurrence matrix
    word_counts = defaultdict(int)
    word_cooccurrences = defaultdict(lambda: defaultdict(int))
    
    # Count word occurrences and co-occurrences
    for doc in processed_docs:
        # Count individual words
        for word in doc:
            word_counts[word] += 1
        
        # Count co-occurrences
        for i, word1 in enumerate(doc):
            for word2 in doc[i+1:]:
                word_cooccurrences[word1][word2] += 1
                word_cooccurrences[word2][word1] += 1
    
    # Calculate PMI-based coherence
    coherence_scores = []
    total_docs = len(processed_docs)
    
    for topic in topic_word_dist:
        topic_words = [word for word, _ in topic]
        topic_score = 0
        pairs = 0
        
        # Calculate PMI for each word pair
        for i, (word1, _) in enumerate(topic):
            for word2, _ in topic[i+1:]:
                if word1 in word_counts and word2 in word_counts:
                    # Calculate probabilities
                    p_w1 = word_counts[word1] / total_docs
                    p_w2 = word_counts[word2] / total_docs
                    p_w1w2 = word_cooccurrences[word1][word2] / total_docs
                    
                    # Calculate PMI if co-occurrence exists
                    if p_w1w2 > 0:
                        pmi = np.log(p_w1w2 / (p_w1 * p_w2))
                        topic_score += pmi
                        pairs += 1
        
        # Average PMI for the topic
        if pairs > 0:
            coherence_scores.append(topic_score / pairs)
    
    # Return average coherence across topics
    return np.mean(coherence_scores) if coherence_scores else 0.0

def exercise_3_topic_similarity(topic1: List[Tuple[str, float]],
                              topic2: List[Tuple[str, float]]) -> float:
    """Solution for Exercise 3: Topic Similarity Measurement"""
    # Convert topics to dictionaries
    dict1 = dict(topic1)
    dict2 = dict(topic2)
    
    # Get all unique words
    all_words = sorted(set(dict1.keys()) | set(dict2.keys()))
    
    # Create probability vectors
    vec1 = np.array([dict1.get(word, 0.0) for word in all_words])
    vec2 = np.array([dict2.get(word, 0.0) for word in all_words])
    
    # Normalize vectors
    vec1 = vec1 / np.sum(vec1)
    vec2 = vec2 / np.sum(vec2)
    
    # Calculate Jensen-Shannon divergence
    m = 0.5 * (vec1 + vec2)
    js_divergence = 0.5 * (entropy(vec1, m) + entropy(vec2, m))
    
    # Convert to similarity score (1 - normalized divergence)
    similarity = 1 - (js_divergence / np.log(2))
    
    return float(similarity)

def exercise_4_dynamic_topics(time_sliced_docs: List[List[str]],
                            num_topics: int = 5) -> List[List[List[Tuple[str, float]]]]:
    """Solution for Exercise 4: Dynamic Topic Modeling"""
    # Initialize vectorizer
    vectorizer = CountVectorizer(max_features=1000)
    
    # Initialize results
    dynamic_topics = []
    prev_model = None
    
    for time_slice in time_sliced_docs:
        # Create document-term matrix
        dtm = vectorizer.fit_transform(time_slice)
        feature_names = vectorizer.get_feature_names_out()
        
        # Create and fit LDA model
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            learning_method='online'
        )
        
        # If there's a previous model, use its components to initialize
        if prev_model is not None:
            lda.components_ = prev_model.components_
        
        lda.fit(dtm)
        
        # Extract topics
        time_slice_topics = []
        for topic_idx, topic in enumerate(lda.components_):
            # Get top words and their probabilities
            top_words = [
                (feature_names[i], float(score))
                for i, score in sorted(
                    enumerate(topic),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            ]
            time_slice_topics.append(top_words)
        
        dynamic_topics.append(time_slice_topics)
        prev_model = lda
    
    return dynamic_topics

def exercise_5_topic_labeling(topic_words: List[List[Tuple[str, float]]]) -> List[str]:
    """Solution for Exercise 5: Automatic Topic Labeling"""
    # Load spaCy for POS tagging and NER
    nlp = spacy.load('en_core_web_sm')
    
    # Function to get compound label from words
    def get_compound_label(words: List[str]) -> str:
        # Process words with spaCy
        doc = nlp(" ".join(words))
        
        # Extract noun phrases
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        if noun_phrases:
            # Use the most relevant noun phrase
            return noun_phrases[0]
        
        # If no noun phrases, combine most common POS patterns
        pos_patterns = []
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN', 'ADJ']:
                pos_patterns.append(token.text)
        
        return " ".join(pos_patterns[:2])
    
    # Generate labels for each topic
    labels = []
    for topic in topic_words:
        # Get top words
        top_words = [word for word, _ in topic[:5]]
        
        # Generate compound label
        label = get_compound_label(top_words)
        
        # Add topic descriptor based on word probabilities
        main_theme = top_words[0]
        if label.lower() not in main_theme.lower():
            label = f"{label} ({main_theme})"
        
        labels.append(label)
    
    return labels

def run_tests():
    """Run tests for the solutions."""
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
    print(f"Topic coherence score: {coherence:.4f}")
    
    # Test Exercise 3
    print("\nTesting Exercise 3: Topic Similarity")
    topic1 = [("machine", 0.1), ("learning", 0.08), ("algorithm", 0.06)]
    topic2 = [("deep", 0.1), ("learning", 0.08), ("neural", 0.06)]
    similarity = exercise_3_topic_similarity(topic1, topic2)
    print(f"Topic similarity score: {similarity:.4f}")
    
    # Test Exercise 4
    print("\nTesting Exercise 4: Dynamic Topics")
    time_slices = [
        documents[:2],
        documents[2:4],
        documents[4:]
    ]
    dynamic_topics = exercise_4_dynamic_topics(time_slices, num_topics=2)
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
    print("Generated topic labels:")
    for idx, label in enumerate(labels):
        print(f"  Topic {idx + 1}: {label}")

if __name__ == "__main__":
    run_tests() 