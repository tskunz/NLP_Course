"""
Examples of working with Lexical Knowledge Bases in Python
This module demonstrates practical usage of various lexical knowledge bases.
"""

from typing import List, Dict, Set, Optional
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import framenet as fn
import spacy
import requests
from collections import defaultdict

class WordNetExplorer:
    """Class for exploring WordNet functionalities"""
    
    def __init__(self):
        """Initialize WordNet explorer"""
        nltk.download('wordnet')
        nltk.download('omw-1.4')
    
    def get_synsets(self, word: str) -> List[Dict]:
        """Get all synsets for a given word"""
        synsets = wn.synsets(word)
        return [
            {
                'name': synset.name(),
                'definition': synset.definition(),
                'examples': synset.examples(),
                'pos': synset.pos()
            }
            for synset in synsets
        ]
    
    def get_semantic_relations(self, word: str) -> Dict[str, Set[str]]:
        """Get semantic relations for a word"""
        relations = defaultdict(set)
        for synset in wn.synsets(word):
            # Hypernyms (is-a relationship)
            relations['hypernyms'].update(
                hyper.lemmas()[0].name() for hyper in synset.hypernyms()
            )
            # Hyponyms (specific types)
            relations['hyponyms'].update(
                hypo.lemmas()[0].name() for hypo in synset.hyponyms()
            )
            # Meronyms (part-of)
            relations['meronyms'].update(
                mero.lemmas()[0].name() for mero in synset.part_meronyms()
            )
            # Holonyms (contains)
            relations['holonyms'].update(
                holo.lemmas()[0].name() for holo in synset.part_holonyms()
            )
        return dict(relations)
    
    def calculate_similarity(self, word1: str, word2: str) -> Optional[float]:
        """Calculate semantic similarity between two words"""
        synsets1 = wn.synsets(word1)
        synsets2 = wn.synsets(word2)
        
        if not synsets1 or not synsets2:
            return None
        
        # Use the first sense of each word
        return synsets1[0].path_similarity(synsets2[0])

class FrameNetExplorer:
    """Class for exploring FrameNet functionalities"""
    
    def __init__(self):
        """Initialize FrameNet explorer"""
        nltk.download('framenet_v17')
    
    def get_frames_for_word(self, word: str) -> List[Dict]:
        """Get all frames associated with a word"""
        frames = []
        for annotation in fn.annotations():
            if word in annotation.text:
                frame = annotation.frame
                frames.append({
                    'frame_name': frame.name,
                    'frame_definition': frame.definition,
                    'frame_elements': [fe.name for fe in frame.FE.values()],
                    'example': annotation.text
                })
        return frames
    
    def explore_frame(self, frame_name: str) -> Dict:
        """Get detailed information about a specific frame"""
        frame = fn.frame(frame_name)
        return {
            'name': frame.name,
            'definition': frame.definition,
            'lexical_units': [lu.name for lu in frame.lexUnit.values()],
            'frame_elements': [fe.name for fe in frame.FE.values()],
            'relations': [rel.name for rel in frame.frameRelations]
        }

class ConceptNetExplorer:
    """Class for exploring ConceptNet"""
    
    def __init__(self):
        """Initialize ConceptNet explorer"""
        self.base_url = "http://api.conceptnet.io"
    
    def get_related_concepts(self, concept: str) -> List[Dict]:
        """Get related concepts from ConceptNet"""
        url = f"{self.base_url}/c/en/{concept}"
        response = requests.get(url)
        if response.status_code != 200:
            return []
        
        data = response.json()
        edges = data.get('edges', [])
        
        return [
            {
                'relation': edge['rel']['label'],
                'start': edge['start']['label'],
                'end': edge['end']['label'],
                'weight': edge.get('weight', 0)
            }
            for edge in edges
        ]
    
    def find_paths(self, concept1: str, concept2: str) -> List[Dict]:
        """Find paths between two concepts"""
        url = f"{self.base_url}/query?node=/c/en/{concept1}&other=/c/en/{concept2}"
        response = requests.get(url)
        if response.status_code != 200:
            return []
        
        data = response.json()
        return [
            {
                'path': [edge['start']['label'] for edge in path['edges']] + 
                       [path['edges'][-1]['end']['label']],
                'weight': sum(edge.get('weight', 0) for edge in path['edges'])
            }
            for path in data.get('paths', [])
        ]

class CustomLexicalKB:
    """Example of building a custom lexical knowledge base"""
    
    def __init__(self):
        """Initialize custom KB with spaCy"""
        self.nlp = spacy.load('en_core_web_sm')
        self.kb = defaultdict(lambda: {
            'definitions': set(),
            'relations': defaultdict(set),
            'examples': set()
        })
    
    def add_entry(self, word: str, definition: str, 
                 relations: Dict[str, List[str]], example: str):
        """Add an entry to the knowledge base"""
        self.kb[word]['definitions'].add(definition)
        for rel_type, related_words in relations.items():
            self.kb[word]['relations'][rel_type].update(related_words)
        self.kb[word]['examples'].add(example)
    
    def extract_relations(self, text: str) -> Dict[str, List[str]]:
        """Extract relations from text using spaCy"""
        doc = self.nlp(text)
        relations = defaultdict(list)
        
        for token in doc:
            if token.dep_ in ('compound', 'amod'):
                relations['modifies'].append((token.text, token.head.text))
            elif token.dep_ == 'nsubj':
                relations['subject_of'].append((token.text, token.head.text))
            elif token.dep_ == 'dobj':
                relations['object_of'].append((token.text, token.head.text))
        
        return dict(relations)
    
    def query(self, word: str) -> Dict:
        """Query the knowledge base"""
        return dict(self.kb[word])

def main():
    """Example usage of lexical knowledge base explorers"""
    
    # WordNet example
    print("\nWordNet Example:")
    wn_explorer = WordNetExplorer()
    word = "computer"
    synsets = wn_explorer.get_synsets(word)
    relations = wn_explorer.get_semantic_relations(word)
    similarity = wn_explorer.calculate_similarity("computer", "machine")
    
    print(f"Synsets for '{word}':")
    for synset in synsets:
        print(f"- {synset['name']}: {synset['definition']}")
    print(f"\nRelations for '{word}':")
    for rel_type, related_words in relations.items():
        print(f"- {rel_type}: {', '.join(related_words)}")
    print(f"\nSimilarity between 'computer' and 'machine': {similarity}")
    
    # FrameNet example
    print("\nFrameNet Example:")
    fn_explorer = FrameNetExplorer()
    frames = fn_explorer.get_frames_for_word("teach")
    print("\nFrames for 'teach':")
    for frame in frames[:2]:  # Show first 2 frames
        print(f"- {frame['frame_name']}: {frame['frame_definition']}")
    
    # ConceptNet example
    print("\nConceptNet Example:")
    cn_explorer = ConceptNetExplorer()
    related = cn_explorer.get_related_concepts("programming")
    print("\nRelated concepts for 'programming':")
    for concept in related[:5]:  # Show first 5 related concepts
        print(f"- {concept['start']} {concept['relation']} {concept['end']}")
    
    # Custom KB example
    print("\nCustom KB Example:")
    custom_kb = CustomLexicalKB()
    
    # Add some entries
    custom_kb.add_entry(
        "python",
        "A high-level programming language",
        {
            "is_a": ["programming_language"],
            "used_for": ["web_development", "data_science"]
        },
        "Python is known for its simple syntax."
    )
    
    # Query the KB
    result = custom_kb.query("python")
    print("\nCustom KB entry for 'python':")
    print(f"- Definition: {next(iter(result['definitions']))}")
    for rel_type, related in result['relations'].items():
        print(f"- {rel_type}: {', '.join(related)}")

if __name__ == "__main__":
    main() 