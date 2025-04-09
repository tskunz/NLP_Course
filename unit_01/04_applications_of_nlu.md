# Applications of Natural Language Understanding (NLU)

## Overview
Natural Language Understanding (NLU) represents the dominant focus in NLP, accounting for approximately 80% of industry applications. This document explores the range of NLU applications, from common use cases to advanced implementations.

## Common Applications

### 1. Automated Text Annotation
- **Tagging**: Generating keywords and phrases for documents
- **Tag Clouds**: Creating visual representations of document content
- **SEO Optimization**: Automating metadata for search engine ranking
```python
def generate_tag_cloud(text: str, max_tags: int = 20):
    """
    Generate a tag cloud from text content
    """
    # Example implementation using NLTK
    from nltk import word_tokenize, pos_tag
    from nltk.corpus import stopwords
    from collections import Counter
    
    # Tokenize and filter
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    
    # Count frequencies
    tag_freq = Counter(tokens)
    
    # Return top tags with weights
    return dict(tag_freq.most_common(max_tags))
```

### 2. Metadata Extraction
- Author identification
- Date extraction
- Copyright information
- Document classification
```python
def extract_metadata(document: str):
    """
    Extract key metadata from document
    """
    metadata = {
        'author': None,
        'date': None,
        'classification': None,
        'copyright': None
    }
    
    # Example patterns for extraction
    import re
    from datetime import datetime
    
    # Date pattern
    date_pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'
    dates = re.findall(date_pattern, document)
    if dates:
        metadata['date'] = datetime.strptime(dates[0], '%m/%d/%Y')
    
    # Copyright pattern
    copyright_pattern = r'©.*\d{4}'
    copyright_matches = re.findall(copyright_pattern, document)
    if copyright_matches:
        metadata['copyright'] = copyright_matches[0]
    
    return metadata
```

### 3. Document Summarization (NLU Approach)
- Key sentence extraction
- Topic word identification
- Representative passage selection
```python
def extract_key_sentences(text: str, num_sentences: int = 2):
    """
    Extract most representative sentences from text
    """
    from nltk.tokenize import sent_tokenize
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Split into sentences
    sentences = sent_tokenize(text)
    
    # Calculate TF-IDF scores
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Score sentences by importance
    sentence_scores = [(i, sum(tfidf_matrix[i].toarray()[0]))
                      for i in range(len(sentences))]
    
    # Get top sentences while preserving order
    top_sentences = sorted(sentence_scores, 
                         key=lambda x: x[1], 
                         reverse=True)[:num_sentences]
    top_sentences = sorted(top_sentences, key=lambda x: x[0])
    
    return [sentences[i] for i, _ in top_sentences]
```

### 4. Corpus Analytics
- Document clustering
- Pattern identification
- Trend analysis
```python
def cluster_documents(documents: list, num_clusters: int = 3):
    """
    Cluster documents based on content similarity
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    
    # Convert documents to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=num_clusters)
    clusters = kmeans.fit_predict(X)
    
    # Group documents by cluster
    clustered_docs = {}
    for i, cluster in enumerate(clusters):
        if cluster not in clustered_docs:
            clustered_docs[cluster] = []
        clustered_docs[cluster].append(documents[i])
    
    return clustered_docs
```

### 5. Taxonomy Mapping
- Cross-taxonomy document classification
- Hierarchical categorization
- Category alignment
```python
class TaxonomyMapper:
    def __init__(self):
        self.source_taxonomy = {}
        self.target_taxonomy = {}
    
    def map_document(self, doc: str, source_category: str):
        """
        Map document from source taxonomy to target taxonomy
        """
        # Example implementation using cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Convert document and target categories to vectors
        vectorizer = TfidfVectorizer()
        doc_vector = vectorizer.fit_transform([doc])
        
        # Find best matching target category
        best_match = None
        best_score = 0
        
        for category, examples in self.target_taxonomy.items():
            category_vectors = vectorizer.transform(examples)
            similarity = cosine_similarity(doc_vector, category_vectors).mean()
            
            if similarity > best_score:
                best_score = similarity
                best_match = category
        
        return best_match
```

### 6. Sentiment Analysis
- Emotion detection
- Opinion mining
- Brand monitoring
```python
def analyze_sentiment(texts: list, brand_name: str = None):
    """
    Analyze sentiment in texts, optionally focusing on specific brand
    """
    from transformers import pipeline
    
    sentiment_analyzer = pipeline("sentiment-analysis")
    
    results = {
        'positive': 0,
        'negative': 0,
        'neutral': 0
    }
    
    for text in texts:
        if brand_name and brand_name.lower() not in text.lower():
            continue
            
        sentiment = sentiment_analyzer(text)[0]
        label = sentiment['label'].lower()
        results[label] += 1
    
    return results
```

## Advanced Applications

### 1. Machine Translation
- Cross-language understanding
- Context preservation
- Cultural adaptation

### 2. Knowledge Discovery
- Fact extraction
- Inference generation
- Truth verification
```python
def verify_fact(claim: str, evidence: list):
    """
    Basic fact verification using textual entailment
    """
    from transformers import pipeline
    
    verifier = pipeline("zero-shot-classification",
                       model="facebook/bart-large-mnli")
    
    # Check if evidence supports or contradicts claim
    labels = ["supports", "contradicts", "neutral"]
    results = []
    
    for e in evidence:
        result = verifier(e, candidate_labels=labels)
        results.append({
            'evidence': e,
            'label': result['labels'][0],
            'confidence': result['scores'][0]
        })
    
    return results
```

### 3. Question Handling
- FAQ mapping
- Query understanding
- Answer generation
```python
def map_to_faq(question: str, faq_list: list):
    """
    Map user question to most relevant FAQ
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Encode question and FAQs
    question_embedding = model.encode([question])
    faq_embeddings = model.encode([q for q, _ in faq_list])
    
    # Find most similar FAQ
    similarities = cosine_similarity(question_embedding, faq_embeddings)[0]
    best_match_idx = similarities.argmax()
    
    return {
        'matched_question': faq_list[best_match_idx][0],
        'answer': faq_list[best_match_idx][1],
        'confidence': similarities[best_match_idx]
    }
```

## Search Applications

### 1. Query Repair
- Typo correction
- Spelling suggestions
```python
def suggest_correction(query: str):
    """
    Suggest corrections for potentially misspelled queries
    """
    from spellchecker import SpellChecker
    
    spell = SpellChecker()
    words = query.split()
    corrections = []
    
    for word in words:
        if word not in spell:
            correction = spell.correction(word)
            if correction != word:
                corrections.append((word, correction))
    
    return corrections
```

### 2. Query Refinement
- Disambiguation
- Query expansion
- Context understanding
```python
def refine_query(query: str):
    """
    Suggest query refinements for ambiguous terms
    """
    ambiguous_terms = {
        'nlp': ['Natural Language Processing', 
                'Neuro-Linguistic Programming'],
        'python': ['Python Programming Language',
                  'Python Snake',
                  'Python Movie'],
        'apple': ['Apple Inc.',
                 'Apple Fruit',
                 'Apple Records']
    }
    
    words = query.lower().split()
    refinements = []
    
    for word in words:
        if word in ambiguous_terms:
            refinements.append({
                'term': word,
                'suggestions': ambiguous_terms[word]
            })
    
    return refinements
```

### 3. Results Post-processing
- Snippet generation
- Result ranking
- Content summarization

## Best Practices
1. **Data Quality**
   - Clean and preprocess input text
   - Handle multiple languages appropriately
   - Account for domain-specific terminology

2. **Performance Optimization**
   - Cache frequent queries/results
   - Use appropriate algorithms for scale
   - Balance accuracy vs. speed

3. **Error Handling**
   - Graceful degradation for edge cases
   - Clear error messages
   - Fallback options

## References
1. Jurafsky, D. & Martin, J.H. "Speech and Language Processing"
2. Manning, C. & Schütze, H. "Foundations of Statistical Natural Language Processing"
3. Bird, S., Klein, E. & Loper, E. "Natural Language Processing with Python"

---
*Note: This document provides an overview of NLU applications with practical examples. The code samples are simplified for illustration and may need additional error handling and optimization for production use.* 