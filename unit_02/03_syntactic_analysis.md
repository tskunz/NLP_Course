# Syntactic Analysis in NLP

## Overview
Syntactic analysis moves beyond individual words to understand how words relate to each other in sentences. This level of analysis is crucial for understanding grammatical structure and relationships between words.

## 1. Sentence Boundary Detection

### 1.1 Challenges
- Periods serve multiple purposes:
  - End of sentence markers
  - Abbreviations (e.g., "Dr.", "St.")
  - Decimal points
  - URLs and email addresses
- Complex cases require context understanding

```python
import nltk
from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer
nltk.download('punkt')

class SentenceBoundaryDetector:
    def __init__(self, custom_abbreviations=None):
        self.tokenizer = PunktSentenceTokenizer()
        if custom_abbreviations:
            self.tokenizer._params.abbrev_types.update(custom_abbreviations)
    
    def detect_sentences(self, text: str) -> list:
        """Split text into sentences handling special cases"""
        # Basic sentence tokenization
        sentences = self.tokenizer.tokenize(text)
        
        # Post-processing for special cases
        refined_sentences = []
        current_sentence = ""
        
        for sent in sentences:
            # Check if sentence was incorrectly split on abbreviation
            if sent.strip().split()[-1].lower() in ['dr', 'mr', 'mrs', 'ms', 'st']:
                current_sentence += sent + " "
            else:
                if current_sentence:
                    refined_sentences.append(current_sentence + sent)
                    current_sentence = ""
                else:
                    refined_sentences.append(sent)
        
        return refined_sentences

# Example usage
detector = SentenceBoundaryDetector()
text = """Dr. Smith visited St. Mary's Hospital. The patient, Mr. Johnson, 
was recovering well. The temp. was 98.6 degrees F. Visit www.hospital.com 
for more info."""

sentences = detector.detect_sentences(text)
for i, sent in enumerate(sentences, 1):
    print(f"Sentence {i}: {sent.strip()}")
```

## 2. Part-of-Speech (POS) Tagging

### 2.1 Penn Treebank POS Tags
Common tags from the Penn Treebank tagset:
- `NN`: Noun, singular
- `NNS`: Noun, plural
- `VB`: Verb, base form
- `VBD`: Verb, past tense
- `JJ`: Adjective
- `RB`: Adverb
- `IN`: Preposition
- `DT`: Determiner

```python
from nltk import pos_tag, word_tokenize
nltk.download('averaged_perceptron_tagger')

class POSTagger:
    def __init__(self):
        self.tag_descriptions = {
            'CC': 'Coordinating conjunction',
            'CD': 'Cardinal number',
            'DT': 'Determiner',
            'EX': 'Existential there',
            'FW': 'Foreign word',
            'IN': 'Preposition/subordinating conjunction',
            'JJ': 'Adjective',
            'JJR': 'Adjective, comparative',
            'JJS': 'Adjective, superlative',
            'NN': 'Noun, singular',
            'NNS': 'Noun, plural',
            'VB': 'Verb, base form',
            'VBD': 'Verb, past tense',
            'VBG': 'Verb, gerund/present participle',
            'VBN': 'Verb, past participle'
        }
    
    def tag_text(self, text: str) -> list:
        """Tag parts of speech in text"""
        words = word_tokenize(text)
        tagged = pos_tag(words)
        
        return [{
            'word': word,
            'tag': tag,
            'description': self.tag_descriptions.get(tag, 'Other')
        } for word, tag in tagged]
    
    def analyze_sentence_structure(self, text: str) -> dict:
        """Analyze sentence structure based on POS tags"""
        tagged = self.tag_text(text)
        
        # Count different parts of speech
        pos_counts = {}
        for item in tagged:
            pos_counts[item['tag']] = pos_counts.get(item['tag'], 0) + 1
        
        # Identify basic sentence components
        has_subject = any(item['tag'].startswith('NN') for item in tagged)
        has_verb = any(item['tag'].startswith('VB') for item in tagged)
        has_object = (has_subject and has_verb and 
                     any(item['tag'].startswith('NN') for item in tagged[1:]))
        
        return {
            'pos_counts': pos_counts,
            'has_subject': has_subject,
            'has_verb': has_verb,
            'has_object': has_object,
            'is_complete': has_subject and has_verb
        }

# Example usage
tagger = POSTagger()
sentence = "The waiter cleared the plates"
analysis = tagger.analyze_sentence_structure(sentence)
tagged_words = tagger.tag_text(sentence)
```

## 3. Syntactic Parsing

### 3.1 Constituency Parsing
Breaking sentences into constituent parts and creating parse trees.

```python
from nltk import Tree
from nltk.parse import CoreNLPParser

class SyntacticParser:
    def __init__(self):
        # Note: Requires Stanford CoreNLP server running
        self.parser = CoreNLPParser(url='http://localhost:9000')
    
    def parse_sentence(self, sentence: str) -> Tree:
        """Generate parse tree for sentence"""
        return next(self.parser.parse(sentence.split()))
    
    def get_noun_phrases(self, tree: Tree) -> list:
        """Extract noun phrases from parse tree"""
        return [' '.join(subtree.leaves()) for subtree in tree.subtrees()
                if subtree.label() == 'NP']
    
    def get_verb_phrases(self, tree: Tree) -> list:
        """Extract verb phrases from parse tree"""
        return [' '.join(subtree.leaves()) for subtree in tree.subtrees()
                if subtree.label() == 'VP']

# Example usage (requires Stanford CoreNLP server)
parser = SyntacticParser()
tree = parser.parse_sentence("John hit the ball")
print(tree)
```

## 4. Lemmatization vs. Stemming

### 4.1 Key Differences
- Stemming: Algorithmic cutting of suffixes
- Lemmatization: Dictionary-based reduction to base form
- Examples:
  - "better" → stem: "bet", lemma: "good"
  - "running" → stem: "run", lemma: "run"

```python
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

class WordNormalizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
    
    def compare_normalization(self, word: str) -> dict:
        """Compare stemming and lemmatization results"""
        return {
            'original': word,
            'stemmed': self.stemmer.stem(word),
            'lemmatized_noun': self.lemmatizer.lemmatize(word, pos='n'),
            'lemmatized_verb': self.lemmatizer.lemmatize(word, pos='v'),
            'lemmatized_adj': self.lemmatizer.lemmatize(word, pos='a')
        }
    
    def analyze_text(self, text: str) -> list:
        """Analyze all words in text"""
        words = word_tokenize(text)
        return [self.compare_normalization(word) for word in words]

# Example usage
normalizer = WordNormalizer()
words = ["better", "running", "lives", "played", "are"]
for word in words:
    result = normalizer.compare_normalization(word)
    print(f"\nAnalysis for '{word}':")
    print(f"Stemmed: {result['stemmed']}")
    print(f"Lemmatized (noun): {result['lemmatized_noun']}")
    print(f"Lemmatized (verb): {result['lemmatized_verb']}")
```

## 5. Discrete Text Field Analysis

### 5.1 Smart ETL (Extract, Transform, Load)
- Extracting structured data from unstructured text
- Normalizing units and formats
- Handling product descriptions and specifications

```python
import re
from typing import Dict, List

class ProductFieldAnalyzer:
    def __init__(self):
        self.patterns = {
            'dimensions': r'(\d+\.?\d*)\s*x\s*(\d+\.?\d*)\s*x\s*(\d+\.?\d*)\s*(inch|cm|mm)',
            'weight': r'(\d+\.?\d*)\s*(kg|g|lbs|oz)',
            'model': r'model[:\s]+([A-Za-z0-9-]+)',
            'processor': r'(Intel|AMD)\s+(Core|Ryzen)\s+([^\s,]+)',
            'memory': r'(\d+)\s*(GB|TB|MB)\s+(RAM|Storage|SSD|HDD)',
        }
    
    def extract_fields(self, text: str) -> Dict[str, List[str]]:
        """Extract structured fields from product description"""
        results = {}
        
        for field, pattern in self.patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            results[field] = [match.group() for match in matches]
        
        return results
    
    def normalize_units(self, value: str, target_unit: str) -> float:
        """Normalize measurements to standard units"""
        # Extract number and current unit
        match = re.match(r'(\d+\.?\d*)\s*([a-zA-Z]+)', value)
        if not match:
            return None
        
        number, unit = float(match.group(1)), match.group(2).lower()
        
        # Conversion factors
        conversions = {
            'length': {
                'mm': 0.001,
                'cm': 0.01,
                'm': 1,
                'inch': 0.0254
            },
            'weight': {
                'g': 0.001,
                'kg': 1,
                'oz': 0.0283495,
                'lbs': 0.453592
            }
        }
        
        # Determine measurement type
        if unit in conversions['length']:
            factors = conversions['length']
        elif unit in conversions['weight']:
            factors = conversions['weight']
        else:
            return None
        
        # Convert to base unit then to target
        base_value = number * factors[unit]
        return base_value / factors[target_unit]
    
    def analyze_product_title(self, title: str) -> dict:
        """Analyze product title for key features"""
        # Extract structured information
        fields = self.extract_fields(title)
        
        # Normalize measurements
        normalized = {}
        for field, values in fields.items():
            if field == 'dimensions':
                normalized[field] = [
                    self.normalize_units(dim, 'cm') 
                    for dim in values
                ]
            elif field == 'weight':
                normalized[field] = [
                    self.normalize_units(w, 'kg') 
                    for w in values
                ]
            else:
                normalized[field] = values
        
        return {
            'raw_fields': fields,
            'normalized': normalized
        }

# Example usage
analyzer = ProductFieldAnalyzer()
product_title = """Acer Aspire E15 Laptop, 15.6" FHD, Intel Core i5-8250U, 
8GB RAM, 256GB SSD, Weight: 5.27 lbs, Dimensions: 15.02 x 10.2 x 1.19 inch"""

analysis = analyzer.analyze_product_title(product_title)
```

## Best Practices

1. **Sentence Boundary Detection**
   - Handle abbreviations carefully
   - Consider domain-specific patterns
   - Use robust tokenization methods

2. **POS Tagging**
   - Use appropriate tagset for your needs
   - Consider context window
   - Handle unknown words gracefully

3. **Parsing**
   - Balance accuracy vs. performance
   - Handle ungrammatical input
   - Consider partial parsing for specific needs

4. **Text Field Analysis**
   - Define clear normalization rules
   - Handle edge cases and exceptions
   - Validate extracted data

## References
1. Marcus, M.P. et al. "Building a Large Annotated Corpus of English: The Penn Treebank"
2. Manning, C. & Schütze, H. "Foundations of Statistical Natural Language Processing"
3. Bird, S., Klein, E. & Loper, E. "Natural Language Processing with Python"

---
*Note: The code examples provided are for educational purposes and may need additional error handling and optimization for production use.* 