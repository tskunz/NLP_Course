# Using Taxonomy Labels for Descriptor-Based Classifiers

## Overview
This section covers how to use hierarchical taxonomy labels in descriptor-based classification systems, particularly useful for content organization and information retrieval.

## Implementation Example

```python
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class TaxonomyNode:
    name: str
    description: str
    parent: Optional['TaxonomyNode'] = None
    children: List['TaxonomyNode'] = None
    vector: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

class TaxonomyClassifier:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')
        self.root = None
        self.nodes = {}
        
    def create_taxonomy(self, taxonomy_dict: Dict):
        """
        Create taxonomy from dictionary structure
        """
        def create_node(name: str, data: Dict, parent: Optional[TaxonomyNode] = None) -> TaxonomyNode:
            description = data.get('description', '')
            node = TaxonomyNode(
                name=name,
                description=description,
                parent=parent,
                vector=self.nlp(description).vector
            )
            self.nodes[name] = node
            
            for child_name, child_data in data.get('children', {}).items():
                child_node = create_node(child_name, child_data, node)
                node.children.append(child_node)
                
            return node
        
        self.root = create_node('root', taxonomy_dict)
        
    def classify_document(self, document: str, threshold: float = 0.5) -> List[Dict]:
        """
        Classify document using taxonomy hierarchy
        """
        doc_vector = self.nlp(document).vector
        results = []
        
        def traverse_taxonomy(node: TaxonomyNode, depth: int = 0):
            similarity = float(cosine_similarity([doc_vector], [node.vector])[0][0])
            
            if similarity >= threshold:
                results.append({
                    'category': node.name,
                    'confidence': similarity,
                    'depth': depth,
                    'path': self._get_path(node)
                })
                
                # Check children if similarity is high enough
                for child in node.children:
                    traverse_taxonomy(child, depth + 1)
                    
        traverse_taxonomy(self.root)
        return sorted(results, key=lambda x: x['confidence'], reverse=True)
    
    def _get_path(self, node: TaxonomyNode) -> List[str]:
        """
        Get path from root to node
        """
        path = []
        current = node
        while current:
            path.append(current.name)
            current = current.parent
        return list(reversed(path))

# Example usage
def create_content_taxonomy():
    taxonomy_dict = {
        'description': 'Root node for content classification',
        'children': {
            'technology': {
                'description': 'Content related to technology and computing',
                'children': {
                    'artificial_intelligence': {
                        'description': 'Topics covering AI, machine learning, and neural networks',
                        'children': {
                            'deep_learning': {
                                'description': 'Deep neural networks, CNN, RNN, and transformers'
                            },
                            'robotics': {
                                'description': 'Robotic systems, automation, and control'
                            }
                        }
                    },
                    'software_development': {
                        'description': 'Software engineering, programming languages, and development methodologies'
                    }
                }
            },
            'science': {
                'description': 'Scientific research and discoveries',
                'children': {
                    'physics': {
                        'description': 'Physical sciences, quantum mechanics, and relativity'
                    },
                    'biology': {
                        'description': 'Life sciences, genetics, and ecology'
                    }
                }
            }
        }
    }
    
    classifier = TaxonomyClassifier()
    classifier.create_taxonomy(taxonomy_dict)
    return classifier

# Advanced Features
class EnhancedTaxonomyClassifier(TaxonomyClassifier):
    def get_related_categories(self, category_name: str, threshold: float = 0.7) -> List[Dict]:
        """
        Find related categories based on description similarity
        """
        target_node = self.nodes.get(category_name)
        if not target_node:
            return []
            
        results = []
        for name, node in self.nodes.items():
            if name != category_name:
                similarity = float(cosine_similarity([target_node.vector], [node.vector])[0][0])
                if similarity >= threshold:
                    results.append({
                        'category': name,
                        'similarity': similarity,
                        'path': self._get_path(node)
                    })
                    
        return sorted(results, key=lambda x: x['similarity'], reverse=True)

## Working with Empty Taxonomies

An empty taxonomy is a hierarchical structure of categories without any example documents or descriptions. For example:
```
News
├── Sports
├── Business
└── Politics
```

## Building Queries from Taxonomy Labels

### 1. Basic Boolean Queries
```python
from typing import List, Dict
import nltk
from nltk.corpus import wordnet as wn

class TaxonomyQueryBuilder:
    def __init__(self):
        self.categories = {}
        
    def create_basic_query(self, target_category: str, other_categories: List[str]) -> str:
        """
        Create a basic Boolean query excluding other categories
        Example: sports AND NOT (business OR politics)
        """
        exclusions = " OR ".join(other_categories)
        return f"{target_category} AND NOT ({exclusions})"
    
    def generate_basic_queries(self, taxonomy: Dict[str, List[str]]) -> List[str]:
        """
        Generate basic Boolean queries for each category
        """
        queries = []
        all_categories = list(taxonomy.keys())
        
        for category in all_categories:
            other_cats = [c for c in all_categories if c != category]
            query = self.create_basic_query(category, other_cats)
            queries.append(query)
            
        return queries
```

### 2. Enhanced Queries with WordNet
```python
class WordNetEnhancedQueryBuilder(TaxonomyQueryBuilder):
    def get_wordnet_expansions(self, word: str) -> List[str]:
        """
        Get synonyms and hypernyms from WordNet
        """
        expansions = set()
        
        # Get synsets
        synsets = wn.synsets(word)
        for synset in synsets:
            # Add synonyms
            expansions.update([lemma.name() for lemma in synset.lemmas()])
            
            # Add hypernyms
            for hypernym in synset.hypernyms():
                expansions.update([lemma.name() for lemma in hypernym.lemmas()])
                
        return list(expansions)
    
    def create_enhanced_query(self, category: str, other_categories: List[str]) -> str:
        """
        Create enhanced query with WordNet expansions
        """
        # Get expansions for target category
        category_terms = self.get_wordnet_expansions(category)
        category_query = " OR ".join(category_terms)
        
        # Get expansions for categories to exclude
        exclusion_terms = []
        for other_cat in other_categories:
            exclusion_terms.extend(self.get_wordnet_expansions(other_cat))
        exclusion_query = " OR ".join(exclusion_terms)
        
        return f"({category_query}) AND NOT ({exclusion_query})"

# Example usage
taxonomy = {
    "sports": [],
    "business": [],
    "politics": []
}

builder = WordNetEnhancedQueryBuilder()
queries = builder.generate_enhanced_queries(taxonomy)
```

## Finding Strong Hits

### 1. Document Scoring
```python
class DocumentScorer:
    def __init__(self, min_score_threshold: float = 0.7):
        self.threshold = min_score_threshold
        
    def calculate_keyword_density(self, document: str, keywords: List[str]) -> float:
        """
        Calculate keyword density in document
        """
        words = document.lower().split()
        keyword_count = sum(1 for word in words if word in keywords)
        return keyword_count / len(words)
    
    def score_document(self, document: str, query_terms: List[str]) -> float:
        """
        Score document based on multiple criteria
        """
        score = 0.0
        
        # Check keyword density
        density = self.calculate_keyword_density(document, query_terms)
        score += density * 0.4
        
        # Check phrase matches
        phrase_matches = self.check_phrase_matches(document, query_terms)
        score += phrase_matches * 0.3
        
        # Check title/metadata importance
        metadata_score = self.check_metadata_importance(document, query_terms)
        score += metadata_score * 0.3
        
        return score
    
    def is_strong_hit(self, document: str, query_terms: List[str]) -> bool:
        """
        Determine if document is a strong hit
        """
        score = self.score_document(document, query_terms)
        return score >= self.threshold
```

## Training Data Generation

### 1. Collecting Training Examples
```python
class TrainingDataCollector:
    def __init__(self, query_builder, document_scorer):
        self.query_builder = query_builder
        self.scorer = document_scorer
        
    def collect_training_data(self, documents: List[str], taxonomy: Dict) -> Dict[str, List[str]]:
        """
        Collect training data for each category
        """
        training_data = {category: [] for category in taxonomy.keys()}
        
        for category in taxonomy.keys():
            # Generate query for category
            query = self.query_builder.create_enhanced_query(
                category,
                [c for c in taxonomy.keys() if c != category]
            )
            
            # Find strong hits
            query_terms = query.split()
            for doc in documents:
                if self.scorer.is_strong_hit(doc, query_terms):
                    training_data[category].append(doc)
                    
        return training_data
```

## Best Practices

1. **Query Construction**
   - Use both synonyms and hypernyms
   - Consider domain-specific terminology
   - Balance query breadth vs. precision

2. **Strong Hit Criteria**
   - Keyword density
   - Phrase matching
   - Title/metadata importance
   - Context relevance

3. **Taxonomy Fit Assessment**
   - Evaluate taxonomy-content alignment
   - Check category coverage
   - Identify potential mismatches

4. **Common Pitfalls**
   - Overly broad queries
   - Insufficient training data
   - Poor taxonomy-content fit
   - Ambiguous category boundaries

## Handling Edge Cases

1. **Insufficient Training Data**
   - Expand WordNet relationships (hypernyms, meronyms)
   - Use domain-specific thesauri
   - Consider manual validation

2. **Category Overlap**
   - Implement stricter exclusion rules
   - Use confidence thresholds
   - Consider hierarchical classification

3. **Poor Taxonomy Fit**
   - Suggest taxonomy modifications
   - Identify alternative data sources
   - Propose hybrid approaches

The success of taxonomy-based classification depends heavily on:
- Quality of initial taxonomy
- Availability of relevant documents
- Appropriate query expansion
- Effective strong hit identification 

## Supplemental Material

### A. Practical Examples

1. **Sports Classification Example**
```python
# Example of expanding sports-related terms
sports_expansions = {
    "sports": ["athletics", "game", "competition"],
    "specific_sports": ["football", "basketball", "baseball", "tennis", "cycling", "skiing"],
    "sports_terms": ["player", "team", "score", "championship", "tournament"],
    "venues": ["stadium", "court", "field", "arena"],
    "roles": ["athlete", "coach", "referee", "player"]
}

# Example query construction
query = """
(sports OR athletics OR game OR competition OR
 football OR basketball OR baseball OR tennis OR
 player OR team OR score OR championship)
AND NOT
(business OR finance OR stock OR trade OR
 politics OR government OR election OR policy)
"""
```

2. **Business Document Example**
```python
# Example of a strong business hit
business_doc = """
The merger between Tech Corp and Innovation Inc was approved by shareholders
today, creating a $50 billion technology giant. The deal, which was first 
announced last quarter, is expected to generate significant synergies in 
R&D and market expansion opportunities.
"""

# Example of a mixed document (business/sports) - would be rejected
mixed_doc = """
The baseball team's new ownership group, led by venture capitalist John Smith,
has invested $200 million in stadium renovations. The group expects the
upgrades to boost annual revenue by 25% through increased ticket sales and
corporate sponsorships.
"""
```

### B. Query Refinement Strategies

1. **Iterative Refinement Process**
```python
class QueryRefiner:
    def __init__(self, initial_query, min_docs=100):
        self.base_query = initial_query
        self.min_docs = min_docs
        
    def refine_query(self, results_count):
        if results_count < self.min_docs:
            # Expand with more hypernyms
            return self.expand_query()
        return self.base_query
        
    def expand_query(self):
        # Add more related terms
        expanded_terms = self.get_additional_terms()
        return f"({self.base_query}) OR ({expanded_terms})"
```

2. **Term Weighting Example**
```python
def calculate_term_importance(term, category_docs):
    """
    Calculate term importance using TF-IDF principles
    """
    term_freq = sum(1 for doc in category_docs if term in doc)
    inverse_category_freq = math.log(len(all_categories) / (1 + category_count_with_term))
    return term_freq * inverse_category_freq
```

### C. Visualization Tools

1. **Taxonomy Visualization**
```python
from graphviz import Digraph

def visualize_taxonomy(taxonomy_dict):
    dot = Digraph(comment='Taxonomy Structure')
    dot.attr(rankdir='TB')
    
    def add_nodes(parent, data):
        for name, info in data.items():
            dot.node(name, name)
            if parent:
                dot.edge(parent, name)
            if 'children' in info:
                add_nodes(name, info['children'])
                
    add_nodes(None, taxonomy_dict)
    return dot

# Usage:
# dot = visualize_taxonomy(taxonomy_dict)
# dot.render('taxonomy_visualization', format='png')
```

### D. Performance Metrics

```python
class TaxonomyPerformanceMetrics:
    def calculate_metrics(self, predictions, true_labels):
        metrics = {
            'coverage': self._calculate_coverage(predictions),
            'precision': self._calculate_precision(predictions, true_labels),
            'hierarchy_score': self._calculate_hierarchy_score(predictions, true_labels)
        }
        return metrics
        
    def _calculate_hierarchy_score(self, predictions, true_labels):
        """
        Calculate how well predictions respect taxonomy hierarchy
        """
        score = 0
        for pred, true in zip(predictions, true_labels):
            common_path = self._longest_common_path(pred, true)
            score += len(common_path) / max(len(pred), len(true))
        return score / len(predictions)
```

### E. Common Challenges and Solutions

1. **Ambiguous Categories**
   - Problem: "Technology" vs "Science" for AI articles
   - Solution: Use hierarchical classification with confidence thresholds

2. **Sparse Data Handling**
   - Problem: Categories with few training examples
   - Solution: Implement few-shot learning techniques

3. **Domain Adaptation**
   - Problem: Taxonomy designed for news applied to academic papers
   - Solution: Domain-specific term expansion and weighting

### F. Additional Resources

1. **Recommended Reading**
   - "Foundations of Statistical Natural Language Processing" (Manning & Schütze)
   - "Information Retrieval" (Manning, Raghavan, & Schütze)
   - "WordNet: An Electronic Lexical Database" (Fellbaum)

2. **Useful Tools and Libraries**
   - NLTK for WordNet integration
   - spaCy for text processing
   - NetworkX for taxonomy manipulation
   - Graphviz for visualization

3. **Online Resources**
   - WordNet online interface
   - Academic paper taxonomies
   - Industry classification standards

### G. Best Practices Checklist

```python
class TaxonomyValidation:
    def validate_taxonomy(self):
        checklist = {
            'structure': self._validate_structure(),
            'coverage': self._validate_coverage(),
            'distinctness': self._validate_distinctness(),
            'balance': self._validate_balance()
        }
        return checklist
        
    def _validate_distinctness(self):
        """
        Check for overlapping categories
        """
        # Implementation details
        pass
```

Remember:
- Start with simple Boolean queries and gradually enhance
- Monitor and adjust query expansion thresholds
- Regularly validate taxonomy fit with content
- Consider domain-specific adjustments
- Document all refinements and their impacts 