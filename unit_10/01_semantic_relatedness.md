# Semantic Analysis: Semantic Relatedness

## Introduction to Semantic Relatedness

Semantic relatedness measures the degree to which two concepts or words are related in meaning. This relationship can be:
- Synonymy (similar meaning)
- Antonymy (opposite meaning)
- Hypernymy/Hyponymy (class/subclass)
- Meronymy/Holonymy (part/whole)
- Thematic relations (contextual associations)

## Word Similarity Measures

### 1. Knowledge-Based Measures
- **Path-based measures**
  - Shortest path in lexical networks
  - Depth-relative path length
  - Weighted edges

- **Information Content (IC) measures**
  - Resnik's measure
  - Lin's measure
  - Jiang-Conrath distance

- **Gloss-based measures**
  - Lesk algorithm
  - Extended gloss overlaps
  - Vector representations of definitions

### 2. Distributional Measures

#### Word Embeddings
```python
# Example using Word2Vec
from gensim.models import Word2Vec

model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)
similarity = model.wv.similarity('computer', 'laptop')
```

#### Types of Word Embeddings
- Word2Vec (CBOW and Skip-gram)
- GloVe (Global Vectors)
- FastText
- BERT embeddings

#### Properties
- Dimensionality reduction
- Contextual relationships
- Analogical reasoning

## Vector Semantics

### 1. Vector Space Models
- Term-Document Matrix
- Term-Term Matrix
- PPMI Matrix
- Dimensionality Reduction Techniques

### 2. Implementation Approaches
```python
def create_term_document_matrix(documents):
    """
    Create a term-document matrix from a collection of documents
    """
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(documents)

def compute_document_similarity(doc1, doc2):
    """
    Compute cosine similarity between documents
    """
    vectors = create_term_document_matrix([doc1, doc2])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
```

### 3. Evaluation Metrics
- Cosine similarity
- Euclidean distance
- Manhattan distance
- Jaccard similarity

## Document Similarity

### 1. Document Representation
- Bag of Words (BoW)
- TF-IDF vectors
- Document embeddings
- Hybrid approaches

### 2. Similarity Computation
```python
def compute_document_embeddings(documents):
    """
    Create document embeddings using sentence transformers
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(documents)

def find_similar_documents(query, documents, embeddings):
    """
    Find documents similar to a query
    """
    query_embedding = compute_document_embeddings([query])[0]
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    return sorted(zip(similarities, documents), reverse=True)
```

### 3. Advanced Techniques
- LSI (Latent Semantic Indexing)
- Doc2Vec
- BERT-based document embeddings
- Cross-encoder models

## Applications of Semantic Similarity

### 1. Information Retrieval
- Semantic search
- Query expansion
- Document ranking
- Relevance feedback

### 2. Text Classification
- K-nearest neighbors
- Semantic clustering
- Topic modeling
- Zero-shot classification

### 3. Question Answering
- Answer selection
- Paraphrase identification
- Entailment recognition
- Fact verification

### 4. Text Summarization
- Sentence similarity
- Redundancy detection
- Content selection
- Summary evaluation

## Best Practices

### 1. Preprocessing
- Text cleaning
- Tokenization
- Lemmatization
- Stop word removal

### 2. Model Selection
- Task requirements
- Data characteristics
- Computational resources
- Interpretability needs

### 3. Evaluation
- Intrinsic evaluation
  - Word similarity benchmarks
  - Analogy tasks
  - Correlation with human judgments

- Extrinsic evaluation
  - Task-specific metrics
  - Cross-validation
  - Error analysis

### 4. Optimization
- Parameter tuning
- Feature selection
- Dimensionality reduction
- Caching strategies

## Common Challenges

1. **Ambiguity**
   - Word sense disambiguation
   - Context-dependent meaning
   - Polysemy and homonymy

2. **Domain Specificity**
   - Domain-specific vocabulary
   - Technical terms
   - Jargon handling

3. **Scalability**
   - Large vocabulary sizes
   - High-dimensional spaces
   - Computational efficiency

4. **Quality Assessment**
   - Ground truth establishment
   - Human evaluation
   - Metric selection

## Future Directions

1. **Contextual Understanding**
   - Better context modeling
   - Cross-lingual semantics
   - Multimodal semantics

2. **Efficiency Improvements**
   - Faster algorithms
   - Reduced memory usage
   - Incremental updates

3. **Advanced Applications**
   - Semantic reasoning
   - Knowledge graph completion
   - Common sense inference

## References

1. Mikolov, T., et al. (2013). Distributed Representations of Words and Phrases and their Compositionality
2. Pennington, J., et al. (2014). GloVe: Global Vectors for Word Representation
3. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers
4. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks

## Additional Resources

- [Word2Vec Tutorial](https://radimrehurek.com/gensim/models/word2vec.html)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [SpaCy's Word Vectors](https://spacy.io/usage/vectors-similarity)
- [HuggingFace Transformers](https://huggingface.co/transformers/) 