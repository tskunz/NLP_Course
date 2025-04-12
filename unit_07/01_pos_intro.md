# Introduction to Part-of-Speech Tagging

## Overview
Part-of-Speech (POS) tagging is a fundamental task in Natural Language Processing that involves marking words in a text with their corresponding part of speech (e.g., noun, verb, adjective). This process is crucial for understanding the grammatical structure of text.

## Parts of Speech

### 1. Universal POS Tags
```python
from typing import List, Dict, Tuple
import spacy

class POSTagger:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.universal_tags = {
            'NOUN': 'Nouns (e.g., book, dog)',
            'VERB': 'Verbs (e.g., run, eat)',
            'ADJ': 'Adjectives (e.g., big, red)',
            'ADV': 'Adverbs (e.g., quickly, very)',
            'PRON': 'Pronouns (e.g., he, they)',
            'DET': 'Determiners (e.g., the, a)',
            'ADP': 'Adpositions (e.g., in, on)',
            'NUM': 'Numerals (e.g., one, 2)',
            'CONJ': 'Conjunctions (e.g., and, but)',
            'PART': 'Particles (e.g., 's, not)'
        }
    
    def tag_text(self, text: str) -> List[Tuple[str, str]]:
        """Tag text with universal POS tags"""
        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc]
    
    def explain_tags(self, tagged_text: List[Tuple[str, str]]) -> List[Dict]:
        """Provide explanations for POS tags"""
        return [{
            'word': word,
            'tag': tag,
            'explanation': self.universal_tags.get(tag, 'Other')
        } for word, tag in tagged_text]
```

### 2. Penn Treebank Tags
- More detailed tagset
- Language-specific distinctions
- Hierarchical relationships

### 3. Custom Tagsets
```python
class CustomPOSTagger:
    def __init__(self):
        self.custom_tags = {
            'N': ['NN', 'NNS', 'NNP', 'NNPS'],
            'V': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
            'ADJ': ['JJ', 'JJR', 'JJS'],
            'ADV': ['RB', 'RBR', 'RBS']
        }
        
    def map_to_custom_tags(self, penn_tags: List[str]) -> List[str]:
        """Map Penn Treebank tags to custom tagset"""
        custom = []
        for tag in penn_tags:
            for custom_tag, penn_list in self.custom_tags.items():
                if tag in penn_list:
                    custom.append(custom_tag)
                    break
            else:
                custom.append('O')  # Other
        return custom
```

## Tagging Schemes

### 1. Rule-Based Tagging
```python
class RuleBasedTagger:
    def __init__(self):
        self.rules = {
            r'\b[A-Z][a-z]+\b': 'NNP',  # Proper nouns
            r'\b\d+\b': 'CD',           # Cardinal numbers
            r'\b[A-Z]+\b': 'NNP',       # Acronyms
            r'\b\w+ing\b': 'VBG',       # Gerunds
            r'\b\w+ed\b': 'VBD'         # Past tense verbs
        }
    
    def apply_rules(self, text: str) -> List[Tuple[str, str]]:
        """Apply rule-based tagging"""
        words = text.split()
        tags = []
        
        for word in words:
            tag = 'NN'  # Default tag
            for pattern, pos_tag in self.rules.items():
                if re.match(pattern, word):
                    tag = pos_tag
                    break
            tags.append((word, tag))
        
        return tags
```

### 2. Statistical Methods
```python
class StatisticalTagger:
    def __init__(self):
        self.tag_probabilities = {}  # P(tag|word)
        self.transition_probs = {}   # P(tag|previous_tag)
        
    def train(self, tagged_corpus: List[List[Tuple[str, str]]]) -> None:
        """Train statistical tagger on corpus"""
        # Count word-tag frequencies
        word_tag_counts = {}
        tag_counts = {}
        
        for sentence in tagged_corpus:
            prev_tag = None
            for word, tag in sentence:
                # Word-tag frequencies
                if word not in word_tag_counts:
                    word_tag_counts[word] = {}
                word_tag_counts[word][tag] = word_tag_counts[word].get(tag, 0) + 1
                
                # Tag frequencies
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
                
                # Transition probabilities
                if prev_tag:
                    if prev_tag not in self.transition_probs:
                        self.transition_probs[prev_tag] = {}
                    self.transition_probs[prev_tag][tag] = \
                        self.transition_probs[prev_tag].get(tag, 0) + 1
                
                prev_tag = tag
        
        # Calculate probabilities
        for word, tags in word_tag_counts.items():
            self.tag_probabilities[word] = {}
            total = sum(tags.values())
            for tag, count in tags.items():
                self.tag_probabilities[word][tag] = count / total
```

## Annotation Standards

### 1. Tokenization
```python
def tokenize_for_tagging(text: str) -> List[str]:
    """Prepare text for POS tagging"""
    # Basic tokenization
    tokens = text.split()
    
    # Handle contractions
    expanded_tokens = []
    for token in tokens:
        if "'" in token:
            # Split contractions (e.g., "don't" -> "do", "n't")
            parts = token.split("'")
            expanded_tokens.extend(parts)
        else:
            expanded_tokens.append(token)
    
    return expanded_tokens
```

### 2. Special Cases
- Compound words
- Multi-word expressions
- Punctuation marks
- Numbers and dates

### 3. Consistency Guidelines
- Annotation rules
- Edge case handling
- Quality control measures

## Best Practices

### 1. Preprocessing
- Text cleaning
- Tokenization
- Normalization

### 2. Model Selection
- Task requirements
- Language specifics
- Performance needs

### 3. Evaluation
- Accuracy metrics
- Error analysis
- Cross-validation

## References
1. Marcus, M. et al. "Building a Large Annotated Corpus of English: The Penn Treebank"
2. Petrov, S. et al. "A Universal Part-of-Speech Tagset"
3. Jurafsky, D. & Martin, J.H. "Speech and Language Processing"

---
*Note: This document introduces POS tagging concepts with practical Python implementations. The code examples are simplified for illustration purposes.* 