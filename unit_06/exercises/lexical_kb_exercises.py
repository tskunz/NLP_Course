"""
Exercises for working with Lexical Knowledge Bases
Complete the following exercises to practice working with various lexical resources.
"""

from typing import List, Dict, Set
import nltk
from nltk.corpus import wordnet as wn

def exercise_1_word_meanings() -> List[Dict]:
    """
    Exercise 1: Word Meanings and Definitions
    
    Task: Implement a function that takes a word and returns all its meanings
    from WordNet, including examples and part of speech.
    
    Example:
    >>> results = exercise_1_word_meanings()
    >>> assert isinstance(results, list)
    >>> assert all(isinstance(item, dict) for item in results)
    >>> assert all('definition' in item for item in results)
    """
    word = "bank"  # Use this word for the exercise
    
    # TODO: Your implementation here
    # 1. Get all synsets for the word
    # 2. For each synset, extract:
    #    - Definition
    #    - Part of speech
    #    - Examples
    #    - Lemma names
    # 3. Return list of dictionaries with the information
    
    return []

def exercise_2_semantic_relations(word: str) -> Dict[str, Set[str]]:
    """
    Exercise 2: Semantic Relations
    
    Task: Implement a function that finds different types of semantic relations
    for a given word using WordNet.
    
    Example:
    >>> relations = exercise_2_semantic_relations("tree")
    >>> assert isinstance(relations, dict)
    >>> assert "synonyms" in relations
    >>> assert "antonyms" in relations
    """
    # TODO: Your implementation here
    # 1. Find synonyms
    # 2. Find antonyms
    # 3. Find hypernyms (is-a relationships)
    # 4. Find hyponyms (types/kinds of)
    # 5. Find meronyms (part-of relationships)
    
    return {
        "synonyms": set(),
        "antonyms": set(),
        "hypernyms": set(),
        "hyponyms": set(),
        "meronyms": set()
    }

def exercise_3_word_similarity(word1: str, word2: str) -> float:
    """
    Exercise 3: Word Similarity
    
    Task: Implement a function that calculates the semantic similarity
    between two words using different WordNet similarity measures.
    
    Example:
    >>> similarity = exercise_3_word_similarity("car", "automobile")
    >>> assert isinstance(similarity, float)
    >>> assert 0 <= similarity <= 1
    """
    # TODO: Your implementation here
    # 1. Get synsets for both words
    # 2. Implement different similarity measures:
    #    - Path similarity
    #    - Wu-Palmer similarity
    #    - Leacock-Chodorow similarity
    # 3. Return the highest similarity score
    
    return 0.0

def exercise_4_custom_kb() -> Dict:
    """
    Exercise 4: Building a Custom Knowledge Base
    
    Task: Implement a simple custom knowledge base for a specific domain
    (e.g., computer science terms).
    
    Example:
    >>> kb = exercise_4_custom_kb()
    >>> assert isinstance(kb, dict)
    >>> assert all(isinstance(v, dict) for v in kb.values())
    """
    # TODO: Your implementation here
    # 1. Create a dictionary-based knowledge base
    # 2. Add entries with:
    #    - Definitions
    #    - Related terms
    #    - Usage examples
    #    - Domain categories
    # 3. Implement basic query functionality
    
    return {}

def exercise_5_relation_extraction(text: str) -> List[Dict]:
    """
    Exercise 5: Relation Extraction
    
    Task: Implement a function that extracts semantic relations
    from a given text using pattern matching or basic NLP techniques.
    
    Example:
    >>> text = "Python is a programming language. It was created by Guido van Rossum."
    >>> relations = exercise_5_relation_extraction(text)
    >>> assert isinstance(relations, list)
    >>> assert all(isinstance(r, dict) for r in relations)
    """
    # TODO: Your implementation here
    # 1. Implement pattern matching for relations like:
    #    - X is a Y
    #    - X was created by Y
    #    - X consists of Y
    # 2. Extract and structure the relations
    # 3. Return list of relation dictionaries
    
    return []

def run_tests():
    """Run tests for the exercises"""
    # Test Exercise 1
    results = exercise_1_word_meanings()
    assert isinstance(results, list), "Exercise 1: Should return a list"
    
    # Test Exercise 2
    relations = exercise_2_semantic_relations("tree")
    assert isinstance(relations, dict), "Exercise 2: Should return a dict"
    assert "synonyms" in relations, "Exercise 2: Should include synonyms"
    
    # Test Exercise 3
    similarity = exercise_3_word_similarity("car", "automobile")
    assert isinstance(similarity, float), "Exercise 3: Should return a float"
    assert 0 <= similarity <= 1, "Exercise 3: Similarity should be between 0 and 1"
    
    # Test Exercise 4
    kb = exercise_4_custom_kb()
    assert isinstance(kb, dict), "Exercise 4: Should return a dict"
    
    # Test Exercise 5
    text = "Python is a programming language. It was created by Guido van Rossum."
    relations = exercise_5_relation_extraction(text)
    assert isinstance(relations, list), "Exercise 5: Should return a list"
    
    print("All test cases passed!")

if __name__ == "__main__":
    # Download required NLTK data
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    
    # Run the tests
    run_tests() 