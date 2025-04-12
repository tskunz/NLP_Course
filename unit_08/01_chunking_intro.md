# Introduction to Text Chunking

## Overview
Text chunking, also known as shallow parsing, is a natural language processing technique that segments text into syntactically correlated parts of words. Unlike full parsing, chunking does not specify the internal structure or the role of chunks in the main sentence.

## Basic Concepts

### 1. Chunk Types
```python
from typing import List, Dict, Tuple
import spacy
import re

class ChunkTypes:
    def __init__(self):
        self.chunk_patterns = {
            'NP': r'<DET>?<ADJ>*<NOUN>+',  # Noun Phrases
            'VP': r'<AUX>?<ADV>*<VERB>+',   # Verb Phrases
            'PP': r'<ADP><NP>',             # Prepositional Phrases
            'ADJP': r'<ADV>*<ADJ>+',        # Adjective Phrases
            'ADVP': r'<ADV>+'               # Adverb Phrases
        }
        
    def explain_pattern(self, chunk_type: str) -> Dict[str, str]:
        """Explain chunk pattern components"""
        return {
            'pattern': self.chunk_patterns.get(chunk_type, ''),
            'explanation': {
                'NP': 'Optional determiner, optional adjectives, required nouns',
                'VP': 'Optional auxiliary, optional adverbs, required verbs',
                'PP': 'Preposition followed by noun phrase',
                'ADJP': 'Optional adverbs followed by adjectives',
                'ADVP': 'One or more adverbs'
            }.get(chunk_type, '')
        }
```

### 2. Phrase Identification
```python
class PhraseIdentifier:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
    def identify_phrases(self, text: str) -> Dict[str, List[str]]:
        """Identify different types of phrases in text"""
        doc = self.nlp(text)
        phrases = {
            'NP': [],  # Noun phrases
            'VP': [],  # Verb phrases
            'PP': [],  # Prepositional phrases
            'ADJP': [], # Adjective phrases
            'ADVP': []  # Adverb phrases
        }
        
        for chunk in doc.noun_chunks:
            phrases['NP'].append(chunk.text)
        
        for token in doc:
            if token.pos_ == 'VERB':
                # Simple verb phrase detection
                phrase = ' '.join([t.text for t in token.subtree 
                                 if t.dep_ in ['aux', 'advmod', 'ROOT']])
                phrases['VP'].append(phrase)
            
            elif token.pos_ == 'ADP':
                # Prepositional phrase detection
                phrase = ' '.join([t.text for t in token.subtree])
                phrases['PP'].append(phrase)
        
        return phrases
```

### 3. Boundary Detection
```python
class BoundaryDetector:
    def __init__(self):
        self.boundary_markers = {
            'start': {
                'NP': ['DET', 'ADJ', 'NOUN'],
                'VP': ['AUX', 'VERB'],
                'PP': ['ADP']
            },
            'end': {
                'NP': ['NOUN', 'PRON'],
                'VP': ['VERB', 'PART'],
                'PP': ['NOUN', 'PRON']
            }
        }
    
    def detect_boundaries(self, tokens: List[Tuple[str, str]]) -> List[Dict]:
        """Detect chunk boundaries in tagged text"""
        boundaries = []
        current_chunk = None
        
        for i, (word, pos) in enumerate(tokens):
            # Check for chunk starts
            for chunk_type, start_pos in self.boundary_markers['start'].items():
                if pos in start_pos and not current_chunk:
                    current_chunk = {
                        'type': chunk_type,
                        'start': i,
                        'words': [word]
                    }
                    break
            
            # Check for chunk continuations
            if current_chunk:
                if pos in self.boundary_markers['end'][current_chunk['type']]:
                    current_chunk['words'].append(word)
                    current_chunk['end'] = i
                    boundaries.append(current_chunk)
                    current_chunk = None
                elif pos not in ['PUNCT', 'CCONJ']:
                    current_chunk['words'].append(word)
        
        return boundaries
```

## Implementation Approaches

### 1. Rule-Based Chunking
```python
class RuleBasedChunker:
    def __init__(self):
        self.rules = [
            # NP rules
            (r'<DET><NOUN>', 'NP'),
            (r'<ADJ><NOUN>', 'NP'),
            (r'<NOUN><NOUN>', 'NP'),
            # VP rules
            (r'<AUX><VERB>', 'VP'),
            (r'<VERB><ADP>', 'VP'),
            # PP rules
            (r'<ADP><DET><NOUN>', 'PP')
        ]
    
    def apply_rules(self, tagged_text: List[Tuple[str, str]]) -> List[Dict]:
        """Apply chunking rules to tagged text"""
        chunks = []
        i = 0
        while i < len(tagged_text) - 1:
            for pattern, chunk_type in self.rules:
                # Convert current sequence to pattern format
                sequence = ''.join(f'<{tag}>' for _, tag in tagged_text[i:i+2])
                if re.match(pattern, sequence):
                    chunks.append({
                        'type': chunk_type,
                        'words': [word for word, _ in tagged_text[i:i+2]]
                    })
                    i += 1
                    break
            i += 1
        return chunks
```

### 2. Statistical Approaches
```python
class StatisticalChunker:
    def __init__(self):
        self.transitions = {}  # P(chunk_tag|prev_chunk_tag)
        self.emissions = {}    # P(pos_tag|chunk_tag)
        
    def train(self, chunked_data: List[Dict]) -> None:
        """Train chunker on annotated data"""
        # Count transitions between chunk types
        for i in range(len(chunked_data) - 1):
            curr_chunk = chunked_data[i]['type']
            next_chunk = chunked_data[i + 1]['type']
            
            if curr_chunk not in self.transitions:
                self.transitions[curr_chunk] = {}
            self.transitions[curr_chunk][next_chunk] = \
                self.transitions[curr_chunk].get(next_chunk, 0) + 1
        
        # Count POS tag emissions for each chunk type
        for chunk in chunked_data:
            chunk_type = chunk['type']
            if chunk_type not in self.emissions:
                self.emissions[chunk_type] = {}
            
            for word in chunk['words']:
                pos_tag = self.get_pos_tag(word)  # Implement POS tagging
                self.emissions[chunk_type][pos_tag] = \
                    self.emissions[chunk_type].get(pos_tag, 0) + 1
```

## Best Practices

### 1. Preprocessing
- POS tagging accuracy
- Text normalization
- Special case handling

### 2. Chunking Strategy
- Task requirements
- Language specifics
- Performance needs

### 3. Evaluation
- Precision and recall
- Boundary accuracy
- Error analysis

## Applications

### 1. Information Extraction
- Named entity recognition
- Relation extraction
- Fact extraction

### 2. Text Summarization
- Key phrase extraction
- Content selection
- Sentence compression

### 3. Machine Translation
- Phrase alignment
- Transfer rules
- Structure mapping

## References
1. Abney, S. "Parsing by Chunks"
2. Ramshaw, L.A. & Marcus, M.P. "Text Chunking using Transformation-Based Learning"
3. Jurafsky, D. & Martin, J.H. "Speech and Language Processing"

---
*Note: This document introduces text chunking concepts with practical Python implementations. The code examples are simplified for illustration purposes.* 