# Introduction to Lexical Knowledge Bases

## Overview
Lexical Knowledge Bases (KBs) are structured repositories of lexical information that provide rich semantic and syntactic information about words and their relationships. They are essential resources for many NLP tasks.

## Basic Concepts

### 1. Structure of Lexical KBs
```python
from typing import Dict, List, Set, Optional

class LexicalEntry:
    def __init__(self, lemma: str):
        self.lemma = lemma
        self.pos_tags = set()
        self.senses = []
        self.relations = {}
        
    def add_sense(self, definition: str, examples: List[str]) -> None:
        """Add a word sense with definition and examples"""
        self.senses.append({
            'definition': definition,
            'examples': examples
        })
    
    def add_relation(self, relation_type: str, target_word: str) -> None:
        """Add a semantic relation to another word"""
        if relation_type not in self.relations:
            self.relations[relation_type] = set()
        self.relations[relation_type].add(target_word)

class LexicalKB:
    def __init__(self):
        self.entries = {}
        
    def add_entry(self, lemma: str) -> LexicalEntry:
        """Add a new lexical entry"""
        if lemma not in self.entries:
            self.entries[lemma] = LexicalEntry(lemma)
        return self.entries[lemma]
    
    def get_entry(self, lemma: str) -> Optional[LexicalEntry]:
        """Retrieve a lexical entry"""
        return self.entries.get(lemma)
```

### 2. Types of Lexical Knowledge
- Morphological information
- Syntactic properties
- Semantic relations
- Usage examples

### 3. Knowledge Organization
```python
class KnowledgeOrganizer:
    def __init__(self):
        self.semantic_relations = {
            'synonymy': set(),
            'antonymy': set(),
            'hypernymy': set(),
            'hyponymy': set(),
            'meronymy': set()
        }
        
    def add_relation(self, word1: str, word2: str, relation_type: str) -> None:
        """Add a semantic relation between words"""
        if relation_type in self.semantic_relations:
            self.semantic_relations[relation_type].add((word1, word2))
    
    def get_related_words(self, word: str, relation_type: str) -> Set[str]:
        """Get words related to the given word by relation type"""
        related = set()
        if relation_type in self.semantic_relations:
            for w1, w2 in self.semantic_relations[relation_type]:
                if w1 == word:
                    related.add(w2)
                elif w2 == word:
                    related.add(w1)
        return related
```

## Components of Lexical KBs

### 1. Vocabulary Management
```python
class VocabularyManager:
    def __init__(self):
        self.vocabulary = set()
        self.frequency = {}
        self.domains = {}
        
    def add_word(self, word: str, domain: str = 'general') -> None:
        """Add a word to the vocabulary"""
        self.vocabulary.add(word)
        self.frequency[word] = self.frequency.get(word, 0) + 1
        
        if domain not in self.domains:
            self.domains[domain] = set()
        self.domains[domain].add(word)
    
    def get_domain_vocabulary(self, domain: str) -> Set[str]:
        """Get vocabulary for a specific domain"""
        return self.domains.get(domain, set())
```

### 2. Semantic Relationships
- Synonymy and antonymy
- Hypernymy and hyponymy
- Meronymy and holonymy
- Cross-references

### 3. Linguistic Properties
```python
class LinguisticProperties:
    def __init__(self):
        self.pos_patterns = {}
        self.subcategorization = {}
        self.selectional_preferences = {}
        
    def add_pos_pattern(self, word: str, pattern: str) -> None:
        """Add part-of-speech pattern for a word"""
        if word not in self.pos_patterns:
            self.pos_patterns[word] = set()
        self.pos_patterns[word].add(pattern)
    
    def add_subcategorization(self, verb: str, frame: str) -> None:
        """Add subcategorization frame for a verb"""
        if verb not in self.subcategorization:
            self.subcategorization[verb] = set()
        self.subcategorization[verb].add(frame)
```

## Applications

### 1. Word Sense Disambiguation
```python
def disambiguate_word(word: str, context: List[str], kb: LexicalKB) -> Optional[Dict]:
    """Simple word sense disambiguation using KB"""
    entry = kb.get_entry(word)
    if not entry or not entry.senses:
        return None
    
    # Simple overlap-based disambiguation
    max_overlap = 0
    best_sense = None
    
    for sense in entry.senses:
        # Count word overlap between context and sense definition/examples
        sense_words = set(sense['definition'].split() + 
                         [w for ex in sense['examples'] for w in ex.split()])
        overlap = len(set(context) & sense_words)
        
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense
    
    return best_sense
```

### 2. Semantic Similarity
```python
def compute_similarity(word1: str, word2: str, kb: LexicalKB) -> float:
    """Compute semantic similarity using KB relations"""
    entry1 = kb.get_entry(word1)
    entry2 = kb.get_entry(word2)
    
    if not entry1 or not entry2:
        return 0.0
    
    # Check direct relations
    for rel_type, targets in entry1.relations.items():
        if word2 in targets:
            return 1.0
    
    # Check shared relations
    shared_relations = 0
    total_relations = 0
    
    for rel_type in entry1.relations:
        if rel_type in entry2.relations:
            shared = len(entry1.relations[rel_type] & 
                        entry2.relations[rel_type])
            shared_relations += shared
            total_relations += len(entry1.relations[rel_type] | 
                                 entry2.relations[rel_type])
    
    return shared_relations / total_relations if total_relations > 0 else 0.0
```

## Best Practices

### 1. Knowledge Base Design
- Modular structure
- Clear relationships
- Efficient access
- Maintainable format

### 2. Quality Control
- Consistency checking
- Coverage assessment
- Regular updates
- Error correction

### 3. Integration Guidelines
- API design
- Access patterns
- Performance optimization
- Documentation

## References
1. Fellbaum, C. "WordNet: An Electronic Lexical Database"
2. Miller, G.A. "WordNet: A Lexical Database for English"
3. Navigli, R. "Word Sense Disambiguation: A Survey"

---
*Note: This document introduces lexical knowledge bases with practical Python implementations. The code examples are simplified for illustration purposes and may need additional features for production use.* 