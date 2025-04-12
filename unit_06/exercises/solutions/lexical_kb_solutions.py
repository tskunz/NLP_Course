"""
Solutions for Lexical Knowledge Base Exercises
This module contains solutions for the lexical KB exercises.
"""

from typing import List, Dict, Set
import nltk
from nltk.corpus import wordnet as wn
import re
from collections import defaultdict

def exercise_1_word_meanings() -> List[Dict]:
    """Solution for Exercise 1: Word Meanings and Definitions"""
    word = "bank"
    results = []
    
    for synset in wn.synsets(word):
        result = {
            'definition': synset.definition(),
            'pos': synset.pos(),
            'examples': synset.examples(),
            'lemmas': [lemma.name() for lemma in synset.lemmas()],
            'name': synset.name()
        }
        results.append(result)
    
    return results

def exercise_2_semantic_relations(word: str) -> Dict[str, Set[str]]:
    """Solution for Exercise 2: Semantic Relations"""
    relations = {
        "synonyms": set(),
        "antonyms": set(),
        "hypernyms": set(),
        "hyponyms": set(),
        "meronyms": set()
    }
    
    for synset in wn.synsets(word):
        # Synonyms and Antonyms
        for lemma in synset.lemmas():
            if lemma.name() != word:
                relations["synonyms"].add(lemma.name())
            for antonym in lemma.antonyms():
                relations["antonyms"].add(antonym.name())
        
        # Hypernyms (is-a relationships)
        for hypernym in synset.hypernyms():
            relations["hypernyms"].update(
                lemma.name() for lemma in hypernym.lemmas()
            )
        
        # Hyponyms (types/kinds of)
        for hyponym in synset.hyponyms():
            relations["hyponyms"].update(
                lemma.name() for lemma in hyponym.lemmas()
            )
        
        # Meronyms (part-of relationships)
        for meronym in synset.part_meronyms():
            relations["meronyms"].update(
                lemma.name() for lemma in meronym.lemmas()
            )
    
    return relations

def exercise_3_word_similarity(word1: str, word2: str) -> float:
    """Solution for Exercise 3: Word Similarity"""
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)
    
    if not synsets1 or not synsets2:
        return 0.0
    
    # Calculate similarities using different measures
    max_similarity = 0.0
    
    for s1 in synsets1:
        for s2 in synsets2:
            # Path similarity
            path_sim = s1.path_similarity(s2) or 0
            
            # Wu-Palmer similarity
            wup_sim = s1.wup_similarity(s2) or 0
            
            # Update max similarity
            max_similarity = max(max_similarity, path_sim, wup_sim)
    
    return max_similarity

def exercise_4_custom_kb() -> Dict:
    """Solution for Exercise 4: Building a Custom Knowledge Base"""
    kb = {
        "python": {
            "definition": "A high-level programming language",
            "related_terms": ["programming", "coding", "software"],
            "examples": [
                "Python is used for web development",
                "Data scientists often use Python"
            ],
            "categories": ["programming_language", "software_development"],
            "attributes": {
                "typing": "dynamic",
                "paradigm": ["object-oriented", "functional", "imperative"],
                "created_by": "Guido van Rossum"
            }
        },
        "algorithm": {
            "definition": "A step-by-step procedure for solving a problem",
            "related_terms": ["computation", "programming", "problem-solving"],
            "examples": [
                "Sorting algorithms arrange data in a specific order",
                "Search algorithms find items in a dataset"
            ],
            "categories": ["computer_science", "mathematics"],
            "attributes": {
                "properties": ["correctness", "efficiency", "termination"],
                "types": ["sorting", "searching", "optimization"]
            }
        },
        "database": {
            "definition": "An organized collection of structured information",
            "related_terms": ["data", "storage", "DBMS"],
            "examples": [
                "MySQL is a popular database system",
                "Databases store and retrieve data efficiently"
            ],
            "categories": ["data_storage", "information_systems"],
            "attributes": {
                "types": ["relational", "NoSQL", "graph"],
                "operations": ["CRUD", "indexing", "querying"]
            }
        }
    }
    
    return kb

def exercise_5_relation_extraction(text: str) -> List[Dict]:
    """Solution for Exercise 5: Relation Extraction"""
    relations = []
    
    # Pattern for "X is a Y"
    is_a_pattern = r'(\w+)\s+is\s+(?:a|an)\s+(\w+)'
    
    # Pattern for "X was created by Y"
    created_by_pattern = r'(\w+)\s+was\s+created\s+by\s+([^.]+)'
    
    # Pattern for "X consists of Y"
    consists_of_pattern = r'(\w+)\s+consists\s+of\s+([^.]+)'
    
    # Find "is a" relations
    for match in re.finditer(is_a_pattern, text):
        relations.append({
            'relation_type': 'is_a',
            'subject': match.group(1),
            'object': match.group(2)
        })
    
    # Find "created by" relations
    for match in re.finditer(created_by_pattern, text):
        relations.append({
            'relation_type': 'created_by',
            'subject': match.group(1),
            'object': match.group(2).strip()
        })
    
    # Find "consists of" relations
    for match in re.finditer(consists_of_pattern, text):
        relations.append({
            'relation_type': 'consists_of',
            'subject': match.group(1),
            'object': match.group(2).strip()
        })
    
    return relations

def main():
    """Run examples of the solutions"""
    # Example 1: Word Meanings
    print("\nExample 1: Word Meanings for 'bank'")
    meanings = exercise_1_word_meanings()
    for i, meaning in enumerate(meanings[:3], 1):  # Show first 3 meanings
        print(f"\n{i}. {meaning['name']}")
        print(f"   Definition: {meaning['definition']}")
        print(f"   Part of Speech: {meaning['pos']}")
        if meaning['examples']:
            print(f"   Example: {meaning['examples'][0]}")
    
    # Example 2: Semantic Relations
    print("\nExample 2: Semantic Relations for 'tree'")
    relations = exercise_2_semantic_relations("tree")
    for rel_type, words in relations.items():
        if words:  # Only show non-empty relations
            print(f"\n{rel_type.capitalize()}:")
            print(", ".join(sorted(list(words))[:5]))  # Show first 5 words
    
    # Example 3: Word Similarity
    print("\nExample 3: Word Similarities")
    word_pairs = [
        ("car", "automobile"),
        ("happy", "joyful"),
        ("computer", "machine")
    ]
    for w1, w2 in word_pairs:
        sim = exercise_3_word_similarity(w1, w2)
        print(f"\nSimilarity between '{w1}' and '{w2}': {sim:.3f}")
    
    # Example 4: Custom Knowledge Base
    print("\nExample 4: Custom Knowledge Base")
    kb = exercise_4_custom_kb()
    for term, info in kb.items():
        print(f"\n{term.capitalize()}:")
        print(f"Definition: {info['definition']}")
        print(f"Categories: {', '.join(info['categories'])}")
    
    # Example 5: Relation Extraction
    print("\nExample 5: Relation Extraction")
    text = """
    Python is a programming language. It was created by Guido van Rossum.
    A computer consists of hardware and software. Java is a programming language.
    """
    relations = exercise_5_relation_extraction(text)
    for relation in relations:
        print(f"\n{relation['relation_type'].replace('_', ' ').title()}:")
        print(f"{relation['subject']} -> {relation['object']}")

if __name__ == "__main__":
    # Download required NLTK data
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    
    # Run the examples
    main() 