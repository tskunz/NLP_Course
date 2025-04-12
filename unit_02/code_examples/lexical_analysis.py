"""
Lexical Analysis Examples
Demonstrates morphological analysis and stemming applications
"""

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import nltk
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class MorphologicalAnalyzer:
    """
    Analyzes morphological structure of words and texts
    """
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def get_wordnet_pos(self, word):
        """Map POS tag to WordNet POS tag"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }
        return tag_dict.get(tag, wordnet.NOUN)
    
    def analyze_word(self, word):
        """Detailed morphological analysis of a single word"""
        pos = self.get_wordnet_pos(word)
        
        analysis = {
            'original': word,
            'stem': self.stemmer.stem(word),
            'lemma': self.lemmatizer.lemmatize(word, pos),
            'pos': pos,
            'features': []
        }
        
        # Check for common morphological features
        if word.endswith('ing'):
            analysis['features'].append('present_participle')
        elif word.endswith('ed'):
            analysis['features'].append('past_tense')
        elif word.endswith('s') and not word.endswith('ss'):
            analysis['features'].append('plural')
        
        # Check for common prefixes
        prefixes = ['un', 're', 'dis', 'pre', 'post', 'anti']
        for prefix in prefixes:
            if word.startswith(prefix):
                analysis['features'].append(f'prefix_{prefix}')
        
        return analysis
    
    def analyze_text(self, text):
        """Analyze morphology of all words in a text"""
        tokens = word_tokenize(text)
        return [self.analyze_word(token) for token in tokens]

def demonstrate_stemming():
    """
    Demonstrate different stemming scenarios
    """
    stemmer = PorterStemmer()
    
    # Example 1: Regular verbs
    verbs = ['run', 'running', 'runs', 'ran']
    print("\nStemming regular verbs:")
    for verb in verbs:
        print(f"{verb:10} -> {stemmer.stem(verb)}")
    
    # Example 2: Irregular verbs
    irregular_verbs = ['go', 'went', 'gone', 'going']
    print("\nStemming irregular verbs:")
    for verb in irregular_verbs:
        print(f"{verb:10} -> {stemmer.stem(verb)}")
    
    # Example 3: Nouns with suffixes
    nouns = ['happiness', 'joyful', 'sadness', 'quickly']
    print("\nStemming nouns with suffixes:")
    for noun in nouns:
        print(f"{noun:10} -> {stemmer.stem(noun)}")

def demonstrate_search_application():
    """
    Demonstrate how stemming helps in search applications
    """
    stemmer = PorterStemmer()
    
    # Sample document database
    documents = {
        'doc1': "The teacher is teaching advanced mathematics",
        'doc2': "Students learn better with interactive teaching methods",
        'doc3': "The school teaches various subjects",
        'doc4': "Education is fundamental for development"
    }
    
    # Create stemmed index
    index = {}
    for doc_id, content in documents.items():
        tokens = word_tokenize(content.lower())
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        for token in stemmed_tokens:
            if token not in index:
                index[token] = set()
            index[token].add(doc_id)
    
    # Search function
    def search(query):
        query_tokens = word_tokenize(query.lower())
        stemmed_query = [stemmer.stem(token) for token in query_tokens]
        
        # Find documents containing any query term
        matching_docs = set()
        for term in stemmed_query:
            if term in index:
                matching_docs.update(index[term])
        
        return {doc_id: documents[doc_id] for doc_id in matching_docs}
    
    # Demonstrate search
    print("\nSearch demonstration:")
    queries = ["teaching", "taught", "teaches"]
    for query in queries:
        print(f"\nSearching for '{query}':")
        results = search(query)
        for doc_id, content in results.items():
            print(f"{doc_id}: {content}")

def main():
    print("=== Lexical Analysis Examples ===")
    
    # Initialize analyzer
    analyzer = MorphologicalAnalyzer()
    
    # Example 1: Analyze individual words
    print("\nMorphological analysis of words:")
    words = ["unhappy", "running", "teachers", "redevelopment"]
    for word in words:
        analysis = analyzer.analyze_word(word)
        print(f"\nWord: {analysis['original']}")
        print(f"Stem: {analysis['stem']}")
        print(f"Lemma: {analysis['lemma']}")
        print(f"POS: {analysis['pos']}")
        print(f"Features: {', '.join(analysis['features'])}")
    
    # Example 2: Demonstrate stemming
    demonstrate_stemming()
    
    # Example 3: Search application
    demonstrate_search_application()

if __name__ == "__main__":
    main() 