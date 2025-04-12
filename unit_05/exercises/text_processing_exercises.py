"""
Text Processing Exercises for NLP
This module contains exercises for practicing text processing techniques.
"""

from typing import List, Dict
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

# Exercise 1: Basic Text Cleaning
def clean_text(text: str) -> str:
    """
    Exercise: Implement basic text cleaning
    
    Tasks:
    1. Convert to lowercase
    2. Remove special characters and numbers
    3. Remove extra whitespace
    4. Remove URLs
    5. Remove email addresses
    
    Example:
    >>> text = "Hello! This is an example... with numbers 123 and email@domain.com"
    >>> clean_text(text)
    'hello this is an example with numbers and email'
    """
    # Your code here
    pass

# Exercise 2: Advanced Tokenization
def custom_tokenize(text: str) -> Dict[str, List]:
    """
    Exercise: Implement custom tokenization
    
    Tasks:
    1. Split into sentences
    2. Split into words
    3. Handle contractions (e.g., "don't" -> "do not")
    4. Preserve important punctuation
    5. Handle special cases (e.g., "U.S.A.")
    
    Return both sentence and word tokens
    
    Example:
    >>> text = "Mr. Smith doesn't like the U.S.A. He prefers Canada!"
    >>> result = custom_tokenize(text)
    >>> print(result['sentences'])
    ["Mr. Smith doesn't like the U.S.A.", "He prefers Canada!"]
    """
    # Your code here
    pass

# Exercise 3: Feature Extraction
def extract_features(text: str) -> Dict:
    """
    Exercise: Extract text features
    
    Tasks:
    1. Calculate word frequencies
    2. Find unique words
    3. Calculate average word length
    4. Identify most common word pairs (bigrams)
    5. Calculate lexical diversity
    
    Example:
    >>> text = "The quick brown fox jumps over the lazy dog"
    >>> features = extract_features(text)
    >>> print(features['unique_words'])
    8
    """
    # Your code here
    pass

# Exercise 4: Text Normalization
def normalize_text(text: str) -> Dict[str, str]:
    """
    Exercise: Implement text normalization
    
    Tasks:
    1. Apply stemming
    2. Apply lemmatization
    3. Remove stopwords
    4. Handle numbers (convert words to digits)
    5. Standardize units (e.g., "km" to "kilometers")
    
    Return original and normalized versions
    
    Example:
    >>> text = "The running dogs were twenty kilometers away"
    >>> result = normalize_text(text)
    >>> print(result['stemmed'])
    'run dog were 20 km away'
    """
    # Your code here
    pass

# Exercise 5: Pattern Matching
def extract_patterns(text: str) -> Dict[str, List[str]]:
    """
    Exercise: Implement pattern matching
    
    Tasks:
    1. Find all email addresses
    2. Find all phone numbers
    3. Find all dates
    4. Find all URLs
    5. Find hashtags and mentions
    
    Example:
    >>> text = "Contact me at user@email.com or call 123-456-7890"
    >>> patterns = extract_patterns(text)
    >>> print(patterns['emails'])
    ['user@email.com']
    """
    # Your code here
    pass

def main():
    """Run exercise examples"""
    # Test text for exercises
    test_text = """
    Hello! This is a sample text for testing NLP exercises. 
    It contains numbers like 123-456-7890 and email@domain.com.
    The quick brown fox jumps over the lazy dog.
    Some people don't like running 20 kilometers!
    Check out https://www.example.com or follow @user #nlp
    """
    
    print("Exercise 1: Text Cleaning")
    print("Original:", test_text)
    print("Cleaned:", clean_text(test_text))
    print()
    
    print("Exercise 2: Custom Tokenization")
    tokens = custom_tokenize(test_text)
    print("Sentences:", tokens.get('sentences', []))
    print("Words:", tokens.get('words', []))
    print()
    
    print("Exercise 3: Feature Extraction")
    features = extract_features(test_text)
    print("Features:", features)
    print()
    
    print("Exercise 4: Text Normalization")
    normalized = normalize_text(test_text)
    print("Original:", normalized.get('original', ''))
    print("Normalized:", normalized.get('normalized', ''))
    print()
    
    print("Exercise 5: Pattern Matching")
    patterns = extract_patterns(test_text)
    print("Found Patterns:", patterns)

if __name__ == "__main__":
    main()

# Solutions will be provided in a separate file 