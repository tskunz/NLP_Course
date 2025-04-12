"""
Solutions for Text Processing Exercises
This module contains solutions for text processing techniques exercises.
"""

from typing import List, Dict
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.util import ngrams
from collections import Counter
import re

# Solution 1: Basic Text Cleaning
def clean_text(text: str) -> str:
    """Basic text cleaning implementation"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Solution 2: Advanced Tokenization
def custom_tokenize(text: str) -> Dict[str, List]:
    """Advanced tokenization implementation"""
    # Handle contractions
    contractions = {
        "n't": " not",
        "'s": " is",
        "'m": " am",
        "'re": " are",
        "'ll": " will",
        "'ve": " have",
        "'d": " would"
    }
    
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Handle special cases
    text = re.sub(r'([A-Z]\.)+', lambda m: m.group(0).replace('.', ''), text)
    
    # Tokenize sentences
    sentences = sent_tokenize(text)
    
    # Tokenize words
    words = []
    for sentence in sentences:
        words.extend(word_tokenize(sentence))
    
    return {
        'sentences': sentences,
        'words': words
    }

# Solution 3: Feature Extraction
def extract_features(text: str) -> Dict:
    """Feature extraction implementation"""
    # Tokenize words
    words = word_tokenize(text.lower())
    
    # Calculate word frequencies
    word_freq = Counter(words)
    
    # Calculate average word length
    avg_word_length = sum(len(word) for word in words) / len(words)
    
    # Get bigrams
    bigrams = list(ngrams(words, 2))
    bigram_freq = Counter(bigrams)
    
    # Calculate lexical diversity
    lexical_diversity = len(set(words)) / len(words)
    
    return {
        'word_frequencies': dict(word_freq),
        'unique_words': len(set(words)),
        'avg_word_length': avg_word_length,
        'common_bigrams': dict(bigram_freq.most_common(5)),
        'lexical_diversity': lexical_diversity
    }

# Solution 4: Text Normalization
def normalize_text(text: str) -> Dict[str, str]:
    """Text normalization implementation"""
    # Initialize tools
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Tokenize
    words = word_tokenize(text.lower())
    
    # Remove stopwords
    filtered_words = [word for word in words if word not in stop_words]
    
    # Apply stemming
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    
    # Apply lemmatization
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    
    # Handle numbers
    number_words = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'twenty': '20', 'thirty': '30', 'forty': '40', 'fifty': '50'
    }
    
    normalized_words = []
    for word in lemmatized_words:
        if word in number_words:
            normalized_words.append(number_words[word])
        else:
            normalized_words.append(word)
    
    return {
        'original': text,
        'filtered': ' '.join(filtered_words),
        'stemmed': ' '.join(stemmed_words),
        'lemmatized': ' '.join(lemmatized_words),
        'normalized': ' '.join(normalized_words)
    }

# Solution 5: Pattern Matching
def extract_patterns(text: str) -> Dict[str, List[str]]:
    """Pattern matching implementation"""
    # Email pattern
    emails = re.findall(r'\S+@\S+\.\S+', text)
    
    # Phone number pattern (various formats)
    phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
    
    # Date pattern (various formats)
    dates = re.findall(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{2,4}', text)
    
    # URL pattern
    urls = re.findall(r'https?://\S+|www\.\S+', text)
    
    # Hashtags and mentions
    hashtags = re.findall(r'#\w+', text)
    mentions = re.findall(r'@\w+', text)
    
    return {
        'emails': emails,
        'phones': phones,
        'dates': dates,
        'urls': urls,
        'hashtags': hashtags,
        'mentions': mentions
    }

def main():
    """Run solution examples"""
    # Test text
    test_text = """
    Hello! This is a sample text for testing NLP exercises. 
    It contains numbers like 123-456-7890 and email@domain.com.
    The quick brown fox jumps over the lazy dog.
    Some people don't like running 20 kilometers!
    Check out https://www.example.com or follow @user #nlp
    Meeting on 15-Dec-2023 at 10/15/2023
    """
    
    print("Solution 1: Text Cleaning")
    print("Original:", test_text)
    print("Cleaned:", clean_text(test_text))
    print()
    
    print("Solution 2: Custom Tokenization")
    tokens = custom_tokenize(test_text)
    print("Sentences:", tokens['sentences'])
    print("Words:", tokens['words'])
    print()
    
    print("Solution 3: Feature Extraction")
    features = extract_features(test_text)
    for key, value in features.items():
        print(f"{key}: {value}")
    print()
    
    print("Solution 4: Text Normalization")
    normalized = normalize_text(test_text)
    for key, value in normalized.items():
        print(f"{key}: {value}")
    print()
    
    print("Solution 5: Pattern Matching")
    patterns = extract_patterns(test_text)
    for key, value in patterns.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main() 