# Basic Text Preprocessing

## Overview
Text preprocessing is a crucial first step in any NLP pipeline. It involves cleaning and standardizing text data to make it suitable for further analysis. This document covers fundamental preprocessing techniques and their implementation.

## Text Cleaning

### 1. Basic Cleaning Operations
```python
import re
from typing import List, Dict

class TextCleaner:
    def __init__(self):
        self.special_chars = re.compile(r'[^a-zA-Z0-9\s]')
        self.multiple_spaces = re.compile(r'\s+')
        
    def clean_text(self, text: str) -> str:
        """Perform basic text cleaning"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = self.special_chars.sub(' ', text)
        
        # Remove extra whitespace
        text = self.multiple_spaces.sub(' ', text)
        
        return text.strip()
    
    def clean_document(self, document: Dict[str, str]) -> Dict[str, str]:
        """Clean document with metadata"""
        return {
            'title': self.clean_text(document.get('title', '')),
            'content': self.clean_text(document.get('content', '')),
            'metadata': document.get('metadata', {})
        }
```

### 2. Character Encoding
- UTF-8 standardization
- ASCII conversion
- Unicode normalization
- Encoding detection

### 3. Special Character Handling
```python
def handle_special_characters(text: str) -> str:
    """Handle various special characters"""
    # Replace common special characters
    replacements = {
        ''': "'",
        '"': '"',
        '"': '"',
        '–': '-',
        '—': '-',
        '…': '...'
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text
```

## Text Standardization

### 1. Case Normalization
```python
def normalize_case(text: str, case: str = 'lower') -> str:
    """Normalize text case"""
    if case == 'lower':
        return text.lower()
    elif case == 'upper':
        return text.upper()
    elif case == 'title':
        return text.title()
    else:
        return text
```

### 2. Number Handling
```python
class NumberNormalizer:
    def __init__(self):
        self.number_words = {
            'one': '1', 'two': '2', 'three': '3',
            'four': '4', 'five': '5', 'six': '6',
            'seven': '7', 'eight': '8', 'nine': '9',
            'zero': '0', 'ten': '10'
        }
        
    def normalize_numbers(self, text: str) -> str:
        """Convert number words to digits"""
        words = text.split()
        normalized = [self.number_words.get(word, word) for word in words]
        return ' '.join(normalized)
    
    def standardize_formats(self, text: str) -> str:
        """Standardize number formats"""
        # Convert decimal formats (1,000.00 -> 1000.00)
        text = re.sub(r'(\d),(\d)', r'\1\2', text)
        
        # Standardize date formats (01-01-2024 -> 2024-01-01)
        text = re.sub(r'(\d{2})-(\d{2})-(\d{4})', r'\3-\1-\2', text)
        
        return text
```

### 3. Punctuation Handling
```python
def handle_punctuation(text: str, mode: str = 'remove') -> str:
    """Handle punctuation in text"""
    if mode == 'remove':
        # Remove all punctuation
        return re.sub(r'[^\w\s]', '', text)
    elif mode == 'standardize':
        # Standardize common punctuation
        text = re.sub(r'[.!?]+', '.', text)  # Standardize sentence endings
        text = re.sub(r'[-_~]', '-', text)   # Standardize hyphens
        text = re.sub(r'[\'"`]', "'", text)  # Standardize quotes
        return text
    else:
        return text
```

## Implementation Considerations

### 1. Performance Optimization
```python
class BatchProcessor:
    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
        self.cleaner = TextCleaner()
        
    def process_large_dataset(self, texts: List[str]) -> List[str]:
        """Process large text datasets in batches"""
        processed_texts = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            processed_batch = [self.cleaner.clean_text(text) for text in batch]
            processed_texts.extend(processed_batch)
        
        return processed_texts
```

### 2. Error Handling
```python
def safe_preprocessing(text: str) -> str:
    """Preprocess text with error handling"""
    try:
        # Handle encoding issues
        if not isinstance(text, str):
            text = str(text, errors='ignore')
        
        # Basic cleaning
        cleaner = TextCleaner()
        text = cleaner.clean_text(text)
        
        # Additional processing
        text = handle_special_characters(text)
        text = handle_punctuation(text, mode='standardize')
        
        return text
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return ""
```

### 3. Quality Assurance
```python
def validate_preprocessing(original: str, processed: str) -> Dict[str, bool]:
    """Validate preprocessing results"""
    return {
        'length_check': len(processed) > 0,
        'case_check': processed.islower(),
        'whitespace_check': not re.search(r'\s{2,}', processed),
        'special_chars_check': not re.search(r'[^a-z0-9\s]', processed)
    }
```

## Best Practices

### 1. Document Your Choices
- Preprocessing steps
- Character handling decisions
- Format standardization rules

### 2. Maintain Consistency
- Use same preprocessing across dataset
- Document any exceptions
- Version control preprocessing code

### 3. Consider Domain Requirements
- Language-specific rules
- Domain-specific terminology
- Special character preservation

## References
1. Manning, C. & Schütze, H. "Foundations of Statistical Natural Language Processing"
2. Bird, S., Klein, E. & Loper, E. "Natural Language Processing with Python"
3. Bengfort, B. et al. "Applied Text Analysis with Python"

---
*Note: This document covers basic text preprocessing techniques with practical Python implementations. The code examples are simplified for illustration purposes and may need additional error handling for production use.* 