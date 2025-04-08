# Descriptor-Based Classifiers

## Overview
Descriptor-based classification involves categorizing documents based on textual descriptions of categories rather than example documents. This approach is particularly useful in legal discovery, FOIA requests, and content filtering scenarios.

## Two-Phase Classification Process

### Phase 1: Information Retrieval
```python
from typing import List, Dict
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class InformationRetriever:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.tokenized_corpus = []
        self.bm25 = None
        
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by tokenizing and removing stop words
        """
        tokens = word_tokenize(text.lower())
        return [token for token in tokens if token not in self.stop_words]
        
    def index_documents(self, documents: List[str]):
        """
        Index documents for BM25 retrieval
        """
        self.tokenized_corpus = [self.preprocess_text(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, float]]:
        """
        Search documents using BM25 scoring
        """
        tokenized_query = self.preprocess_text(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k documents
        top_indices = np.argsort(doc_scores)[-top_k:][::-1]
        return [
            {'doc_id': idx, 'score': doc_scores[idx]}
            for idx in top_indices
            if doc_scores[idx] > 0
        ]
```

### Phase 2: Content-Based Classification
```python
class StrongHitAnalyzer:
    def __init__(self, min_keyword_density: float = 0.02):
        self.min_density = min_keyword_density
        
    def calculate_keyword_density(self, text: str, keywords: List[str]) -> float:
        """
        Calculate keyword density in text
        """
        words = text.lower().split()
        keyword_count = sum(1 for word in words if word in keywords)
        return keyword_count / len(words)
        
    def analyze_phrase_matches(self, text: str, phrases: List[str]) -> Dict[str, int]:
        """
        Analyze exact phrase matches
        """
        matches = {}
        for phrase in phrases:
            matches[phrase] = text.lower().count(phrase.lower())
        return matches
        
    def is_strong_hit(self, document: str, query_info: Dict) -> bool:
        """
        Determine if a document is a strong hit based on multiple criteria
        """
        # Check keyword density
        density = self.calculate_keyword_density(
            document, 
            query_info['keywords']
        )
        if density < self.min_density:
            return False
            
        # Check phrase matches
        phrase_matches = self.analyze_phrase_matches(
            document,
            query_info['key_phrases']
        )
        if not any(count > 0 for count in phrase_matches.values()):
            return False
            
        # Check section presence (if required)
        if query_info.get('required_sections'):
            for section in query_info['required_sections']:
                if section.lower() not in document.lower():
                    return False
                    
        return True
```

## Advanced Implementation Strategies

### 1. Query Expansion with Domain Knowledge
```python
class DomainAwareQueryExpander:
    def __init__(self, domain_ontology: Dict[str, List[str]]):
        self.ontology = domain_ontology
        
    def expand_query(self, base_query: str) -> str:
        """
        Expand query using domain-specific terminology
        """
        expanded_terms = []
        query_terms = base_query.lower().split()
        
        for term in query_terms:
            # Add base term
            expanded_terms.append(term)
            
            # Add domain-specific related terms
            if term in self.ontology:
                expanded_terms.extend(self.ontology[term])
                
        return " OR ".join(f'"{term}"' for term in expanded_terms)

# Example usage
legal_ontology = {
    "patent": ["intellectual property", "IP rights", "patent claims"],
    "infringement": ["violation", "unauthorized use", "infringes"],
    "technology": ["technical implementation", "system architecture"]
}

expander = DomainAwareQueryExpander(legal_ontology)
expanded_query = expander.expand_query("patent infringement technology")
```

### 2. Contextual Relevance Scoring
```python
class ContextualScorer:
    def __init__(self, context_weights: Dict[str, float]):
        self.weights = context_weights
        
    def score_document_context(self, document: str, query_context: Dict) -> float:
        """
        Score document based on contextual relevance
        """
        score = 0.0
        
        # Time relevance
        if 'time_period' in query_context:
            time_score = self.evaluate_time_relevance(
                document,
                query_context['time_period']
            )
            score += time_score * self.weights['time']
            
        # Source authority
        if 'source_types' in query_context:
            source_score = self.evaluate_source_authority(
                document,
                query_context['source_types']
            )
            score += source_score * self.weights['source']
            
        # Document type relevance
        if 'doc_types' in query_context:
            type_score = self.evaluate_doc_type_match(
                document,
                query_context['doc_types']
            )
            score += type_score * self.weights['type']
            
        return score
```

## Real-World Example: Brewing Process Classification

```python
def create_brewing_classifier():
    classifier = DescriptorClassifier()
    
    # Add detailed brewing process categories
    classifier.add_category(
        "mashing",
        """
        Documents describing the process of mixing crushed malted grains with hot water
        to convert starches into fermentable sugars. Including temperature control,
        enzyme activity, pH levels, and mash schedules. Key terms: mash tun,
        saccharification, beta-amylase, alpha-amylase, diastatic power, conversion.
        """
    )
    
    classifier.add_category(
        "fermentation",
        """
        Content related to yeast activity, sugar conversion to alcohol, fermentation
        temperature control, and monitoring. Including specific gravity readings,
        attenuation levels, yeast health, and fermentation vessel specifications.
        Key aspects: primary fermentation, secondary fermentation, yeast pitching,
        krausen formation, and final gravity targets.
        """
    )
    
    return classifier

# Example document
brewing_doc = """
During today's brew session, we maintained the mash at 152°F for 60 minutes,
achieving full conversion as verified by the iodine test. The wort was collected
and the specific gravity reading showed 1.048, indicating good efficiency in
extracting fermentable sugars from the grain bill.
"""
```

## Best Practices for Descriptor Writing

1. **Clarity and Specificity**
   - Use precise, unambiguous language
   - Include measurable criteria
   - Define scope boundaries

2. **Comprehensiveness**
   - Cover all relevant aspects
   - Include variations and synonyms
   - Consider edge cases

3. **Structure and Organization**
   - Use consistent formatting
   - Group related concepts
   - Prioritize key elements

4. **Maintenance and Updates**
   - Regular review cycles
   - Version control
   - Change documentation

## Performance Optimization

1. **Query Processing**
```python
class QueryOptimizer:
    def __init__(self):
        self.cache = {}
        
    def optimize_query(self, query: str) -> str:
        """
        Optimize query for performance
        """
        # Check cache
        if query in self.cache:
            return self.cache[query]
            
        # Remove unnecessary terms
        optimized = self.remove_low_value_terms(query)
        
        # Reorder for efficiency
        optimized = self.reorder_terms(optimized)
        
        # Cache result
        self.cache[query] = optimized
        return optimized
```

2. **Batch Processing**
```python
def process_document_batch(documents: List[str], batch_size: int = 100):
    """
    Process documents in batches for better performance
    """
    results = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_results = process_batch(batch)
        results.extend(batch_results)
    return results
```

## Quality Metrics

1. **Classification Accuracy**
```python
def evaluate_classifier(classifier, test_cases: List[Dict]):
    metrics = {
        'precision': [],
        'recall': [],
        'f1_score': []
    }
    
    for case in test_cases:
        results = classifier.classify_document(case['document'])
        
        # Calculate metrics
        precision = calculate_precision(results, case['expected'])
        recall = calculate_recall(results, case['expected'])
        f1 = 2 * (precision * recall) / (precision + recall)
        
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1)
        
    return {k: np.mean(v) for k, v in metrics.items()}
```

Remember:
- Start with clear, well-defined descriptors
- Use domain-specific terminology
- Implement strong hit criteria
- Monitor and adjust performance
- Regularly update and maintain descriptors

```python
import spacy
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class DescriptorClassifier:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")  # Using larger model for better semantics
        self.categories = {}
        
    def add_category(self, name: str, description: str):
        """
        Add a category with its descriptor
        """
        # Process the description and store its vector
        doc = self.nlp(description)
        self.categories[name] = {
            'description': description,
            'vector': doc.vector
        }
        
    def classify_document(self, document: str, threshold: float = 0.5) -> List[Dict[str, float]]:
        """
        Classify a document against all category descriptors
        """
        doc = self.nlp(document)
        results = []
        
        for category_name, category_data in self.categories.items():
            similarity = cosine_similarity(
                [doc.vector],
                [category_data['vector']]
            )[0][0]
            
            if similarity >= threshold:
                results.append({
                    'category': category_name,
                    'confidence': float(similarity)
                })
                
        return sorted(results, key=lambda x: x['confidence'], reverse=True)

# Example usage
def legal_discovery_example():
    classifier = DescriptorClassifier()
    
    # Add category descriptors
    classifier.add_category(
        "patent_infringement",
        """
        Documents discussing the use, implementation, or development of technology 
        related to neural network architectures for image recognition, specifically 
        focusing on convolutional neural networks and their application in mobile devices.
        This includes technical specifications, meeting notes, email discussions, 
        and development plans from 2018-2023.
        """
    )
    
    classifier.add_category(
        "trade_secrets",
        """
        Communications and documents containing confidential information about 
        proprietary algorithms, data processing techniques, and internal research 
        findings related to machine learning optimization methods. Including 
        documentation of novel approaches, experimental results, and performance metrics.
        """
    )
    
    # Test document
    document = """
    In yesterday's meeting, we discussed the implementation of our new CNN architecture
    for mobile devices. The team presented benchmark results showing a 15% improvement
    in inference speed while maintaining accuracy above 95%. Attached are the technical
    specifications and performance comparisons with competing solutions.
    """
    
    results = classifier.classify_document(document, threshold=0.4)
    return results

# Advanced Features
class EnhancedDescriptorClassifier(DescriptorClassifier):
    def extract_key_phrases(self, text: str) -> List[str]:
        """
        Extract important phrases from text
        """
        doc = self.nlp(text)
        phrases = []
        
        for chunk in doc.noun_chunks:
            if not chunk.root.is_stop:
                phrases.append(chunk.text)
                
        return phrases
    
    def analyze_semantic_overlap(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Analyze semantic similarity between texts
        """
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        
        # Get key phrases
        phrases1 = self.extract_key_phrases(text1)
        phrases2 = self.extract_key_phrases(text2)
        
        # Calculate overlap
        common_phrases = set(phrases1).intersection(set(phrases2))
        
        return {
            'similarity': doc1.similarity(doc2),
            'common_phrases': list(common_phrases),
            'unique_to_first': list(set(phrases1) - set(phrases2)),
            'unique_to_second': list(set(phrases2) - set(phrases1))
        }
```

## Key Components

1. **Category Description Processing**
   - Semantic parsing
   - Key phrase extraction
   - Vector representation

2. **Similarity Calculation**
```python
def calculate_similarity_scores(doc_vector, category_vectors):
    """
    Calculate similarity scores between a document and categories
    """
    similarities = cosine_similarity([doc_vector], category_vectors)
    return similarities[0]
```

## Best Practices

1. **Description Writing**
   - Be specific and detailed
   - Use domain-specific terminology
   - Include variations and synonyms

2. **Threshold Selection**
   - Balance precision and recall
   - Consider domain requirements
   - Use validation data

3. **Performance Optimization**
   - Cache category vectors
   - Batch processing
   - Parallel computation

4. **Quality Assurance**
   - Regular description updates
   - Validation against experts
   - Performance monitoring 