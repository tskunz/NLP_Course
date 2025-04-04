# Descriptor-Based Classifiers

## Overview
Descriptor-based classification involves categorizing documents based on textual descriptions of categories rather than example documents. This approach is particularly useful in legal discovery, FOIA requests, and content filtering scenarios.

## Implementation Example

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