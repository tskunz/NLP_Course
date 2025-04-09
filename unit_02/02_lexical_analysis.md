# Lexical Analysis: Morphology and Stemming

## Overview
Lexical analysis forms the foundation of Natural Language Processing, dealing with the basic units of language - words and morphemes. This analysis is crucial for understanding how words are formed and how they can be reduced to their basic forms.

## 1. Core Concepts

### 1.1 Lexicon
- A machine-readable dictionary
- Contains base forms of words
- May include additional information like:
  - Part of speech
  - Word frequency
  - Semantic relationships

### 1.2 Morphology
- Study of word formation and internal structure
- Deals with morphemes (smallest meaningful units)
- Examples:
  - Root words: "run", "play", "teach"
  - Affixes: "-ing", "-ed", "-er", "un-", "re-"

### 1.3 Stemming
- Process of reducing words to their root/stem form
- Removes affixes (prefixes and suffixes)
- Examples:
  - running → run
  - played → play
  - teacher → teach
  - unhappy → happy

## 2. Practical Implementation

### 2.1 Basic Stemming Example
```python
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

class MorphologyAnalyzer:
    def __init__(self):
        self.stemmer = PorterStemmer()
    
    def analyze_word(self, word: str) -> dict:
        """Analyze morphological structure of a word"""
        return {
            'original': word,
            'stemmed': self.stemmer.stem(word),
            'is_plural': word.endswith('s'),  # Simple plural check
            'has_ing': word.endswith('ing'),
            'has_ed': word.endswith('ed')
        }
    
    def analyze_text(self, text: str) -> list:
        """Analyze morphology of all words in a text"""
        tokens = word_tokenize(text)
        return [self.analyze_word(token) for token in tokens]

# Example usage
analyzer = MorphologyAnalyzer()
text = "The teachers are running multiple educational programs"
analysis = analyzer.analyze_text(text)
for word_analysis in analysis:
    print(f"Word: {word_analysis['original']}")
    print(f"Stem: {word_analysis['stemmed']}")
    print("Morphological features:", end=" ")
    features = []
    if word_analysis['is_plural']: features.append("plural")
    if word_analysis['has_ing']: features.append("present participle")
    if word_analysis['has_ed']: features.append("past tense/participle")
    print(", ".join(features) if features else "no additional features")
    print()
```

### 2.2 Advanced Morphological Analysis
```python
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class AdvancedMorphologyAnalyzer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
    
    def get_wordnet_pos(self, word: str) -> str:
        """Map POS tag to WordNet POS tag"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }
        return tag_dict.get(tag, wordnet.NOUN)
    
    def analyze_morphology(self, word: str) -> dict:
        """Detailed morphological analysis"""
        pos = self.get_wordnet_pos(word)
        lemma = self.lemmatizer.lemmatize(word, pos)
        
        # Identify common affixes
        prefixes = ['un', 're', 'dis', 'pre']
        suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'tion', 'ment']
        
        found_prefixes = [p for p in prefixes if word.startswith(p)]
        found_suffixes = [s for s in suffixes if word.endswith(s)]
        
        return {
            'original': word,
            'lemma': lemma,
            'pos': pos,
            'prefixes': found_prefixes,
            'suffixes': found_suffixes,
            'is_complex': bool(found_prefixes or found_suffixes)
        }

# Example usage
advanced_analyzer = AdvancedMorphologyAnalyzer()
words = ["unhappy", "rewriting", "disconnected", "preparation", 
         "quickly", "strongest", "development"]

for word in words:
    analysis = advanced_analyzer.analyze_morphology(word)
    print(f"\nAnalysis for '{analysis['original']}':")
    print(f"Base form (lemma): {analysis['lemma']}")
    print(f"Part of Speech: {analysis['pos']}")
    if analysis['prefixes']:
        print(f"Prefixes found: {', '.join(analysis['prefixes'])}")
    if analysis['suffixes']:
        print(f"Suffixes found: {', '.join(analysis['suffixes'])}")
```

## 3. Applications of Stemming

### 3.1 Search Engines
```python
class SearchEngine:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.document_base = {}
    
    def add_document(self, doc_id: str, content: str):
        """Add a document to the search engine"""
        # Tokenize and stem all words
        tokens = word_tokenize(content.lower())
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Store both original and stemmed versions
        self.document_base[doc_id] = {
            'original': content,
            'stemmed_tokens': stemmed_tokens
        }
    
    def search(self, query: str) -> list:
        """Search for documents matching the query"""
        # Stem the query words
        query_tokens = word_tokenize(query.lower())
        stemmed_query = [self.stemmer.stem(token) for token in query_tokens]
        
        results = []
        for doc_id, doc in self.document_base.items():
            # Check if any query terms match in stemmed form
            if any(term in doc['stemmed_tokens'] for term in stemmed_query):
                results.append({
                    'doc_id': doc_id,
                    'content': doc['original'],
                    'relevance': sum(term in doc['stemmed_tokens'] 
                                   for term in stemmed_query)
                })
        
        # Sort by relevance
        return sorted(results, key=lambda x: x['relevance'], reverse=True)

# Example usage
search_engine = SearchEngine()

# Add some documents
documents = {
    'doc1': "Running in the morning is great exercise",
    'doc2': "The runner completed the marathon",
    'doc3': "She runs every day",
    'doc4': "The race was exciting"
}

for doc_id, content in documents.items():
    search_engine.add_document(doc_id, content)

# Search for "running"
results = search_engine.search("running")
print("\nSearch results for 'running':")
for result in results:
    print(f"Document {result['doc_id']}: {result['content']}")
    print(f"Relevance score: {result['relevance']}")
```

### 3.2 Text Classification
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

class TextClassifier:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.vectorizer = CountVectorizer(
            preprocessor=self.preprocess_text
        )
        self.classifier = MultinomialNB()
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text using stemming"""
        tokens = word_tokenize(text.lower())
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        return ' '.join(stemmed_tokens)
    
    def train(self, texts: list, labels: list):
        """Train the classifier"""
        # Convert texts to feature vectors
        X = self.vectorizer.fit_transform(texts)
        # Train the classifier
        self.classifier.fit(X, labels)
    
    def predict(self, text: str) -> str:
        """Predict the category of a text"""
        # Preprocess and vectorize the text
        X = self.vectorizer.transform([text])
        # Make prediction
        return self.classifier.predict(X)[0]

# Example usage
classifier = TextClassifier()

# Training data
training_texts = [
    "I am running in the park",
    "The program is running slowly",
    "She runs a successful business",
    "The computer runs many programs",
    "Running shoes are on sale"
]
training_labels = [
    "physical_activity",
    "computing",
    "business",
    "computing",
    "physical_activity"
]

# Train the classifier
classifier.train(training_texts, training_labels)

# Test the classifier
test_texts = [
    "He is running a marathon",
    "The server is running multiple processes"
]

for text in test_texts:
    category = classifier.predict(text)
    print(f"\nText: {text}")
    print(f"Predicted category: {category}")
```

## 4. Best Practices

### 4.1 Stemming Considerations
- Choose appropriate stemming algorithm
  - Porter Stemmer: Most common, aggressive
  - Lancaster Stemmer: Very aggressive
  - Snowball Stemmer: More accurate but slower
- Handle special cases and exceptions
- Consider language-specific requirements

### 4.2 Performance Optimization
- Cache stemmed words
- Use batch processing for large datasets
- Consider memory vs. speed tradeoffs

## 5. Common Applications
1. **Search Engines**
   - Query expansion
   - Index optimization
   - Relevance matching

2. **Text Classification**
   - Document categorization
   - Spam filtering
   - Content tagging

3. **Information Retrieval**
   - Document indexing
   - Content matching
   - Keyword extraction

4. **Text Analytics**
   - Content analysis
   - Trend detection
   - Pattern recognition

## 6. Advanced Lexical Analysis

### 6.1 Corpus-Derived Metadata
- Statistical information derived from large text collections
- Word frequency analysis
- Co-occurrence patterns
- Example applications:
  - Word clouds
  - Topic modeling
  - Term importance scoring

```python
from collections import Counter
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

class CorpusAnalyzer:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def extract_metadata(self, documents: list) -> dict:
        """Extract metadata from a corpus of documents"""
        # Initialize counters
        word_freq = Counter()
        word_pairs = Counter()
        
        for doc in documents:
            # Tokenize and stem
            tokens = word_tokenize(doc.lower())
            stemmed = [self.stemmer.stem(t) for t in tokens 
                      if t not in self.stop_words]
            
            # Count individual words
            word_freq.update(stemmed)
            
            # Count word pairs (co-occurrences)
            for i in range(len(stemmed)-1):
                pair = (stemmed[i], stemmed[i+1])
                word_pairs.update([pair])
        
        return {
            'word_frequencies': dict(word_freq),
            'cooccurrences': dict(word_pairs),
            'vocabulary_size': len(word_freq),
            'total_words': sum(word_freq.values())
        }
    
    def generate_word_cloud_data(self, word_freq: dict, 
                               min_freq: int = 2) -> dict:
        """Generate data suitable for word cloud visualization"""
        return {word: freq for word, freq in word_freq.items() 
                if freq >= min_freq}

# Example usage
analyzer = CorpusAnalyzer()
corpus = [
    "The moral implications of AI development are significant",
    "Morality in technology raises ethical questions",
    "AI systems must be developed with strong moral principles",
    "The morals of machine learning affect society"
]

metadata = analyzer.extract_metadata(corpus)
word_cloud_data = analyzer.generate_word_cloud_data(
    metadata['word_frequencies']
)
```

### 6.2 Collocations and Multi-word Expressions
- Words that commonly occur together
- Different meaning when combined
- Examples:
  - "take care" vs "take" + "care"
  - "ping pong" vs "ping" + "pong"
  - "hot dog" vs "hot" + "dog"

```python
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

class CollocationAnalyzer:
    def __init__(self):
        self.measures = BigramAssocMeasures()
    
    def find_collocations(self, text: str, n_best: int = 10) -> list:
        """Find significant word collocations in text"""
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Find bigram collocations
        finder = BigramCollocationFinder.from_words(tokens)
        
        # Apply frequency filter
        finder.apply_freq_filter(2)
        
        # Score collocations using PMI (Pointwise Mutual Information)
        return finder.nbest(self.measures.pmi, n_best)
    
    def build_collocation_lexicon(self, texts: list) -> dict:
        """Build a lexicon of common collocations"""
        lexicon = {}
        
        for text in texts:
            collocations = self.find_collocations(text)
            for col in collocations:
                key = ' '.join(col)
                if key not in lexicon:
                    lexicon[key] = {
                        'frequency': 1,
                        'components': col
                    }
                else:
                    lexicon[key]['frequency'] += 1
        
        return lexicon

# Example usage
collocation_analyzer = CollocationAnalyzer()
texts = [
    "He needs to take care of his health",
    "Please take care of this matter",
    "Taking care of business is important",
    "She takes good care of her family"
]

lexicon = collocation_analyzer.build_collocation_lexicon(texts)
```

### 6.3 Polysemy and Word Sense Disambiguation
- Words with multiple meanings
- Context-dependent interpretation
- WordNet integration for sense lookup

```python
from nltk.corpus import wordnet as wn

class WordSenseAnalyzer:
    def __init__(self):
        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()
    
    def get_word_senses(self, word: str) -> list:
        """Get all possible senses of a word from WordNet"""
        senses = []
        for synset in wn.synsets(word):
            senses.append({
                'definition': synset.definition(),
                'examples': synset.examples(),
                'pos': synset.pos(),
                'lemma_names': synset.lemma_names()
            })
        return senses
    
    def disambiguate_by_domain(self, word: str, 
                             context: str, 
                             domain: str) -> dict:
        """
        Simple domain-based word sense disambiguation
        """
        # Get all senses
        senses = self.get_word_senses(word)
        
        # Define domain keywords
        domain_keywords = {
            'sports': {'team', 'player', 'game', 'score', 'field'},
            'finance': {'money', 'bank', 'cost', 'price', 'market'},
            'technology': {'computer', 'software', 'digital', 'data'}
        }
        
        # Tokenize context
        context_words = set(word_tokenize(context.lower()))
        
        # Score each sense based on domain overlap
        best_score = 0
        best_sense = None
        
        for sense in senses:
            # Get words from definition and examples
            sense_words = set(word_tokenize(sense['definition'].lower()))
            for example in sense['examples']:
                sense_words.update(word_tokenize(example.lower()))
            
            # Calculate domain relevance score
            domain_words = domain_keywords.get(domain, set())
            score = len(sense_words & domain_words)
            
            if score > best_score:
                best_score = score
                best_sense = sense
        
        return {
            'word': word,
            'domain': domain,
            'selected_sense': best_sense,
            'confidence_score': best_score
        }

# Example usage
sense_analyzer = WordSenseAnalyzer()

# Example with the word "table"
word = "table"
contexts = [
    ("The database table contains customer records.", "technology"),
    ("Please clear the table after dinner.", "household"),
    ("Let's table this discussion for now.", "business")
]

for context, domain in contexts:
    result = sense_analyzer.disambiguate_by_domain(word, context, domain)
    print(f"\nContext: {context}")
    print(f"Domain: {domain}")
    print(f"Selected sense: {result['selected_sense']['definition']}")
```

### 6.4 Applications of Advanced Lexical Analysis

#### 1. Spell Correction
```python
from difflib import get_close_matches

class SpellChecker:
    def __init__(self, word_list_file: str = None):
        self.word_set = set(nltk.corpus.words.words())
        if word_list_file:
            with open(word_list_file) as f:
                self.word_set.update(f.read().splitlines())
    
    def correct_word(self, word: str) -> str:
        """Suggest correction for a potentially misspelled word"""
        if word in self.word_set:
            return word
        
        # Find close matches
        matches = get_close_matches(word, self.word_set, n=1)
        return matches[0] if matches else word
    
    def correct_text(self, text: str) -> str:
        """Correct spelling in entire text"""
        words = word_tokenize(text)
        corrected = [self.correct_word(word) for word in words]
        return ' '.join(corrected)

# Example usage
spell_checker = SpellChecker()
text = "The comuter is procesing data"
corrected = spell_checker.correct_text(text)
```

#### 2. Terminology Extraction
```python
from sklearn.feature_extraction.text import TfidfVectorizer

class TerminologyExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_df=0.95,  # Remove very common words
            min_df=2,     # Remove very rare words
            stop_words='english'
        )
    
    def extract_terms(self, documents: list, top_n: int = 10) -> list:
        """Extract key terminology from documents"""
        # Calculate TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        
        # Get feature names (terms)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Calculate average TF-IDF score for each term
        avg_scores = tfidf_matrix.mean(axis=0).A1
        
        # Sort terms by score
        term_scores = list(zip(feature_names, avg_scores))
        term_scores.sort(key=lambda x: x[1], reverse=True)
        
        return term_scores[:top_n]

# Example usage
extractor = TerminologyExtractor()
documents = [
    "AI systems process large amounts of data",
    "Machine learning algorithms require significant processing power",
    "Data processing is essential for AI applications"
]

key_terms = extractor.extract_terms(documents)
```

#### 3. Lexical Diversity Analysis
```python
class LexicalDiversityAnalyzer:
    def __init__(self):
        self.stemmer = PorterStemmer()
    
    def calculate_diversity(self, text: str) -> dict:
        """Calculate various lexical diversity metrics"""
        # Tokenize and normalize
        tokens = word_tokenize(text.lower())
        stems = [self.stemmer.stem(token) for token in tokens 
                if token.isalnum()]
        
        # Calculate metrics
        total_words = len(stems)
        unique_words = len(set(stems))
        
        return {
            'total_words': total_words,
            'unique_words': unique_words,
            'type_token_ratio': unique_words / total_words if total_words > 0 else 0,
            'vocabulary_density': unique_words / (total_words ** 0.5) 
                                if total_words > 0 else 0
        }
    
    def compare_texts(self, texts: dict) -> dict:
        """Compare lexical diversity across multiple texts"""
        results = {}
        for name, text in texts.items():
            results[name] = self.calculate_diversity(text)
        return results

# Example usage
diversity_analyzer = LexicalDiversityAnalyzer()
texts = {
    'text1': "The quick brown fox jumps over the lazy dog",
    'text2': "The dog runs and runs and runs in circles"
}

comparison = diversity_analyzer.compare_texts(texts)
```

## Best Practices for Advanced Lexical Analysis

1. **Quality of Lexical Resources**
   - Start with established resources (WordNet)
   - Extend for domain-specific needs
   - Regularly update and maintain custom lexicons

2. **Domain Awareness**
   - Consider context and domain in disambiguation
   - Build domain-specific collocation lists
   - Use domain experts for validation

3. **Performance Optimization**
   - Cache frequent lookups
   - Use efficient data structures
   - Implement batch processing

4. **Error Handling**
   - Handle unknown words gracefully
   - Provide fallback mechanisms
   - Log and analyze failures

## References
1. Fellbaum, C. "WordNet: An Electronic Lexical Database"
2. Manning, C. & Schütze, H. "Foundations of Statistical Natural Language Processing"
3. Bird, S., Klein, E. & Loper, E. "Natural Language Processing with Python"

---
*Note: The code examples provided are for educational purposes and may need additional error handling and optimization for production use.* 