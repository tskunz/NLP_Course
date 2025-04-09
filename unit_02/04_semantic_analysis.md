# Semantic Analysis in NLP

## Overview
Semantic analysis is the process of understanding the meaning and interpretation of words, signs, and sentence structure. Unlike lexical and syntactic analysis that focus on form, semantic analysis deals with meaning and context.

## Major Types of Semantic Analysis

### 1. Named-Entity Recognition (NER/NEE)

#### Academic vs. Real-World Challenges
- **Academic NER**: 
  - Focuses on formal text (e.g., Wikipedia articles)
  - Handles standard naming conventions
  - Example: "Hillary Rodham Clinton" → "Hillary Clinton", "Secretary Clinton"

- **Real-World NER**:
  - Must handle informal references and slang
  - Deals with intentional misspellings and nicknames
  - Requires context-aware algorithms
  - Example: Identifying "Hitlery" as referring to "Hillary Clinton"

```python
from nltk import ne_chunk, pos_tag, word_tokenize
from typing import List, Dict

class RealWorldNER:
    def __init__(self):
        self.name_variants = {}
        self.context_memory = {}
    
    def learn_name_variant(self, text: str) -> None:
        """Learn new name variants from context"""
        # Example: Learning from patterns like "Hillary Clinton... Hitlery"
        tokens = word_tokenize(text)
        for i, token in enumerate(tokens):
            if i > 0 and tokens[i-1] in self.name_variants:
                # Context-based variant learning
                self.context_memory[token] = self.name_variants[tokens[i-1]]
    
    def identify_entities(self, text: str) -> List[Dict[str, str]]:
        """Identify named entities including informal variants"""
        entities = []
        tokens = word_tokenize(text)
        
        # Basic NER using NLTK
        chunks = ne_chunk(pos_tag(tokens))
        
        # Enhance with learned variants
        for chunk in chunks:
            if hasattr(chunk, 'label'):
                entity = {
                    'text': ' '.join([c[0] for c in chunk]),
                    'type': chunk.label(),
                    'variants': self.context_memory.get(chunk[0][0], [])
                }
                entities.append(entity)
        
        return entities
```

### 2. Relationship Extraction

Relationship extraction identifies connections between named entities by understanding:
- Entity types (Person, Organization, Location)
- Relationship types (CEO, Employee, Founder)
- Ontological constraints

Example:
```python
class RelationshipExtractor:
    def __init__(self):
        self.ontology = {
            'Person': {
                'can_be': ['CEO', 'Employee', 'Founder'],
                'cannot_be': ['Subsidiary', 'Product']
            },
            'Organization': {
                'can_have': ['CEO', 'Employees', 'Products'],
                'cannot_have': ['Birthday', 'Age']
            }
        }
    
    def extract_relationship(self, text: str) -> Dict:
        """Extract relationships between entities"""
        # Example: "Tim Cook is the CEO of Apple"
        relationships = {
            'subject': {'text': 'Tim Cook', 'type': 'Person'},
            'relationship': 'CEO',
            'object': {'text': 'Apple', 'type': 'Organization'}
        }
        
        # Validate against ontology
        if relationships['relationship'] in self.ontology[relationships['subject']['type']]['can_be']:
            return relationships
        return None
```

### 3. Word Sense Disambiguation (WSD)

The process of identifying which sense of a word is used in a sentence. Example using the word "chair":

```python
from nltk.corpus import wordnet as wn

class WordSenseDisambiguator:
    def __init__(self):
        self.context_words = {
            'furniture': {'sit', 'table', 'room', 'comfortable'},
            'academic': {'professor', 'department', 'university', 'endowed'},
            'meeting': {'committee', 'preside', 'moderate', 'session'}
        }
    
    def disambiguate(self, word: str, context: str) -> str:
        """Determine the sense of a word based on context"""
        context_set = set(word_tokenize(context.lower()))
        
        # Score each sense based on context overlap
        scores = {}
        for sense, context_words in self.context_words.items():
            score = len(context_set.intersection(context_words))
            scores[sense] = score
        
        return max(scores.items(), key=lambda x: x[1])[0]
```

### 4. Classification and Taxonomy

#### Challenges in Document Classification:
- Multiple classification possibilities
- Overlapping categories
- Non-strict taxonomies

Popular taxonomies:
- DMOZ/Curlie (Web directory)
- IAB (Internet Advertising)
- IMDB (Movie genres)
- Product taxonomies (Amazon, Google Shopping)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

class DocumentClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.classifier = LinearSVC()
        self.taxonomy = {
            'technology': ['software', 'hardware', 'internet'],
            'sports': ['football', 'basketball', 'soccer'],
            'entertainment': ['movies', 'music', 'games']
        }
    
    def train(self, documents: List[str], labels: List[str]) -> None:
        """Train the classifier"""
        X = self.vectorizer.fit_transform(documents)
        self.classifier.fit(X, labels)
    
    def predict(self, document: str) -> List[str]:
        """Predict document categories"""
        X = self.vectorizer.transform([document])
        return self.classifier.predict(X)
```

### 5. Topic Segmentation and Sentiment Analysis

Combined example showing multiple semantic analyses:

```python
class SemanticAnalyzer:
    def __init__(self):
        self.sentiment_words = {
            'positive': {'good', 'excellent', 'improved', 'successful'},
            'negative': {'bad', 'poor', 'unfortunate', 'failed'}
        }
    
    def analyze_document(self, text: str) -> Dict:
        """Perform multiple semantic analyses on a document"""
        paragraphs = text.split('\n\n')
        
        analysis = {
            'tags': self.extract_tags(text),
            'segments': self.segment_topics(paragraphs),
            'sentiment': self.analyze_sentiment(text)
        }
        
        return analysis
    
    def extract_tags(self, text: str) -> List[str]:
        """Extract key topics as tags"""
        # Implementation here
        pass
    
    def segment_topics(self, paragraphs: List[str]) -> List[Dict]:
        """Identify topic shifts in the document"""
        # Implementation here
        pass
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment throughout the document"""
        # Implementation here
        pass
```

## Best Practices

1. **Named Entity Recognition**
   - Build robust context-aware systems
   - Handle informal and variant references
   - Maintain updated entity databases

2. **Relationship Extraction**
   - Develop comprehensive ontologies
   - Validate relationships against domain constraints
   - Consider temporal aspects of relationships

3. **Word Sense Disambiguation**
   - Use wide context windows
   - Consider domain-specific senses
   - Leverage multiple disambiguation techniques

4. **Classification**
   - Clean and normalize taxonomies
   - Handle multi-label classification
   - Consider hierarchical relationships

5. **Topic Segmentation and Sentiment**
   - Use multiple indicators for topic shifts
   - Consider context in sentiment analysis
   - Combine different semantic analyses for richer insights

## References
1. Jurafsky, D. & Martin, J.H. "Speech and Language Processing"
2. Manning, C.D. & Schütze, H. "Foundations of Statistical Natural Language Processing"
3. Bird, S., Klein, E. & Loper, E. "Natural Language Processing with Python" 