"""
Examples of Parsing Techniques in NLP
This module demonstrates various parsing approaches including chunking,
constituency parsing, and dependency parsing.
"""

from typing import List, Dict, Tuple, Optional
import nltk
from nltk import pos_tag, word_tokenize, RegexpParser
from nltk.tree import Tree
from nltk.parse import ChartParser
from nltk.parse.generate import generate
import spacy
from collections import defaultdict

class ChunkParser:
    """Implements chunking (shallow parsing) functionality"""
    
    def __init__(self):
        """Initialize the chunk parser"""
        # Download required NLTK data
        nltk.download('averaged_perceptron_tagger')
        nltk.download('punkt')
        
        # Define chunk grammar
        self.grammar = r"""
            NP: {<DT>?<JJ.*>*<NN.*>+}     # Noun phrase
            VP: {<VB.*><NP|PP>}           # Verb phrase
            PP: {<IN><NP>}                # Prepositional phrase
            ADJP: {<JJ.*>}                # Adjective phrase
            ADVP: {<RB.*>}                # Adverb phrase
        """
        self.parser = RegexpParser(self.grammar)
    
    def parse(self, text: str) -> Tree:
        """Parse text into chunks"""
        # Tokenize and POS tag
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        
        # Apply chunking
        return self.parser.parse(tagged)
    
    def extract_phrases(self, text: str) -> Dict[str, List[str]]:
        """Extract different types of phrases from text"""
        tree = self.parse(text)
        phrases = defaultdict(list)
        
        for subtree in tree.subtrees():
            if subtree.label() in ['NP', 'VP', 'PP', 'ADJP', 'ADVP']:
                phrase = ' '.join(word for word, tag in subtree.leaves())
                phrases[subtree.label()].append(phrase)
        
        return dict(phrases)

class ConstituencyParser:
    """Implements constituency (full grammar) parsing"""
    
    def __init__(self):
        """Initialize the constituency parser"""
        # Define a simple CFG
        self.grammar = nltk.CFG.fromstring("""
            S -> NP VP
            NP -> Det N | N | Det N PP
            VP -> V NP | V PP | V NP PP
            PP -> P NP
            Det -> 'the' | 'a'
            N -> 'cat' | 'dog' | 'park'
            V -> 'chased' | 'saw' | 'walked'
            P -> 'in' | 'to' | 'with'
        """)
        
        self.parser = ChartParser(self.grammar)
    
    def parse(self, text: str) -> List[Tree]:
        """Parse text using the CFG"""
        tokens = word_tokenize(text.lower())
        return list(self.parser.parse(tokens))
    
    def generate_sentences(self, n: int = 5) -> List[str]:
        """Generate grammatical sentences from the CFG"""
        sentences = []
        for sentence in generate(self.grammar, n=n):
            sentences.append(' '.join(sentence))
        return sentences
    
    def draw_tree(self, text: str):
        """Draw the parse tree for visualization"""
        trees = self.parse(text)
        if trees:
            trees[0].draw()

class DependencyParser:
    """Implements dependency parsing using spaCy"""
    
    def __init__(self):
        """Initialize the dependency parser"""
        self.nlp = spacy.load('en_core_web_sm')
    
    def parse(self, text: str) -> List[Dict[str, str]]:
        """Parse text and return dependency relations"""
        doc = self.nlp(text)
        dependencies = []
        
        for token in doc:
            dependencies.append({
                'word': token.text,
                'dep': token.dep_,
                'head': token.head.text,
                'head_pos': token.head.pos_,
                'children': [child.text for child in token.children]
            })
        
        return dependencies
    
    def extract_subject_object_pairs(self, text: str) -> List[Dict[str, str]]:
        """Extract subject-verb-object triples from text"""
        doc = self.nlp(text)
        triples = []
        
        for token in doc:
            if token.dep_ == "ROOT":
                subject = None
                direct_object = None
                
                # Find subject and object
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subject = child.text
                    elif child.dep_ == "dobj":
                        direct_object = child.text
                
                if subject or direct_object:
                    triples.append({
                        'subject': subject,
                        'verb': token.text,
                        'object': direct_object
                    })
        
        return triples

class HybridParser:
    """Combines multiple parsing approaches"""
    
    def __init__(self):
        """Initialize the hybrid parser"""
        self.chunker = ChunkParser()
        self.dependency_parser = DependencyParser()
    
    def parse(self, text: str) -> Dict:
        """Perform hybrid parsing"""
        # Get chunks
        chunks = self.chunker.extract_phrases(text)
        
        # Get dependencies
        dependencies = self.dependency_parser.parse(text)
        
        # Combine analyses
        return {
            'chunks': chunks,
            'dependencies': dependencies,
            'subject_object_pairs': self.dependency_parser.extract_subject_object_pairs(text)
        }
    
    def analyze_sentence_structure(self, text: str) -> Dict:
        """Perform detailed sentence structure analysis"""
        doc = self.dependency_parser.nlp(text)
        
        analysis = {
            'sentence_type': self._determine_sentence_type(doc),
            'main_verb': self._find_main_verb(doc),
            'phrases': self.chunker.extract_phrases(text),
            'complexity': self._analyze_complexity(doc)
        }
        
        return analysis
    
    def _determine_sentence_type(self, doc) -> str:
        """Determine the type of sentence"""
        # Check for question marks
        if doc.text.endswith('?'):
            return 'question'
        
        # Check for exclamation marks
        if doc.text.endswith('!'):
            return 'exclamation'
        
        # Check for imperative (command)
        first_token = doc[0]
        if first_token.pos_ == 'VERB':
            return 'imperative'
        
        return 'declarative'
    
    def _find_main_verb(self, doc) -> Optional[str]:
        """Find the main verb of the sentence"""
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                return token.text
        return None
    
    def _analyze_complexity(self, doc) -> Dict:
        """Analyze sentence complexity"""
        return {
            'n_tokens': len(doc),
            'n_clauses': len([token for token in doc if token.dep_ == "ROOT"]),
            'has_subordinate': any(token.dep_ == "ccomp" for token in doc),
            'has_relative_clause': any(token.dep_ == "relcl" for token in doc)
        }

def main():
    """Demonstrate usage of parsers"""
    text = "The black cat chased the small mouse in the park"
    
    print("\n1. Chunk Parsing Example:")
    chunker = ChunkParser()
    chunks = chunker.extract_phrases(text)
    print("\nExtracted phrases:")
    for phrase_type, phrases in chunks.items():
        print(f"\n{phrase_type}:")
        for phrase in phrases:
            print(f"  - {phrase}")
    
    print("\n2. Constituency Parsing Example:")
    constituency_parser = ConstituencyParser()
    simple_text = "the cat chased the dog"
    trees = constituency_parser.parse(simple_text)
    print("\nGenerated sentences:")
    for sentence in constituency_parser.generate_sentences(3):
        print(f"  - {sentence}")
    
    print("\n3. Dependency Parsing Example:")
    dependency_parser = DependencyParser()
    deps = dependency_parser.parse(text)
    print("\nDependency relations:")
    for dep in deps:
        print(f"\nWord: {dep['word']}")
        print(f"Relation: {dep['dep']}")
        print(f"Head: {dep['head']}")
        print(f"Children: {', '.join(dep['children'])}")
    
    print("\n4. Hybrid Parsing Example:")
    hybrid_parser = HybridParser()
    analysis = hybrid_parser.analyze_sentence_structure(text)
    print("\nSentence analysis:")
    print(f"Type: {analysis['sentence_type']}")
    print(f"Main verb: {analysis['main_verb']}")
    print("\nComplexity metrics:")
    for metric, value in analysis['complexity'].items():
        print(f"  {metric}: {value}")

if __name__ == "__main__":
    main() 