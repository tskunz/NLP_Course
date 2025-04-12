"""
Syntactic Analysis Examples
Demonstrates various aspects of syntactic analysis in NLP
"""

import re
import nltk
from typing import List, Dict, Tuple, Optional
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

class SyntacticAnalyzer:
    """
    A class for performing various syntactic analysis tasks on text.
    """
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        # Common abbreviations that might be confused with sentence boundaries
        self.common_abbrev = {'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'sr.', 'jr.', 'etc.', 'e.g.',
                            'i.e.', 'vs.', 'fig.', 'sq.', 'lbs.', 'in.'}
        
    def detect_sentence_boundaries(self, text: str) -> List[str]:
        """
        Detects sentence boundaries while handling special cases.
        
        Args:
            text: Input text to be analyzed
            
        Returns:
            List of detected sentences
        """
        try:
            # Pre-process to handle special cases
            # Temporarily replace decimal numbers
            text = re.sub(r'(\d+)\.(\d+)', r'\1<DECIMAL>\2', text)
            
            # Handle known abbreviations
            for abbrev in self.common_abbrev:
                text = text.replace(abbrev, abbrev.replace('.', '<PERIOD>'))
            
            # Detect sentences
            sentences = sent_tokenize(text)
            
            # Restore original text
            sentences = [s.replace('<DECIMAL>', '.').replace('<PERIOD>', '.') 
                        for s in sentences]
            
            return sentences
            
        except Exception as e:
            print(f"Error in sentence boundary detection: {str(e)}")
            return [text]  # Return original text as single sentence on error
    
    def analyze_pos_tags(self, sentence: str) -> List[Dict[str, str]]:
        """
        Performs detailed POS tagging with lemmatization.
        
        Args:
            sentence: Input sentence to analyze
            
        Returns:
            List of dictionaries containing word analysis
        """
        try:
            tokens = word_tokenize(sentence)
            pos_tags = pos_tag(tokens)
            
            analysis = []
            for word, tag in pos_tags:
                word_info = {
                    'word': word,
                    'pos': tag,
                    'lemma': self.lemmatizer.lemmatize(word),
                    'is_content_word': self._is_content_word(tag)
                }
                analysis.append(word_info)
                
            return analysis
            
        except Exception as e:
            print(f"Error in POS analysis: {str(e)}")
            return []
            
    def analyze_sentence_structure(self, sentence: str) -> Dict[str, List[str]]:
        """
        Identifies basic sentence components (subject, verb, object).
        
        Args:
            sentence: Input sentence to analyze
            
        Returns:
            Dictionary with sentence components
        """
        try:
            analysis = self.analyze_pos_tags(sentence)
            structure = {
                'subjects': [],
                'verbs': [],
                'objects': []
            }
            
            for i, word_info in enumerate(analysis):
                pos = word_info['pos']
                
                # Identify subjects (nouns preceded by determiners)
                if pos.startswith('NN'):
                    if i > 0 and analysis[i-1]['pos'] == 'DT':
                        structure['subjects'].append(word_info['word'])
                
                # Identify verbs
                elif pos.startswith('VB'):
                    structure['verbs'].append(word_info['word'])
                
                # Identify objects (nouns following verbs)
                elif pos.startswith('NN') and i > 0 and analysis[i-1]['pos'].startswith('VB'):
                    structure['objects'].append(word_info['word'])
            
            return structure
            
        except Exception as e:
            print(f"Error in sentence structure analysis: {str(e)}")
            return {'subjects': [], 'verbs': [], 'objects': []}
    
    def _is_content_word(self, pos_tag: str) -> bool:
        """
        Determines if a word is a content word based on its POS tag.
        """
        content_pos = {'NN', 'NNS', 'NNP', 'NNPS',  # Nouns
                      'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # Verbs
                      'JJ', 'JJR', 'JJS',  # Adjectives
                      'RB', 'RBR', 'RBS'}  # Adverbs
        return pos_tag in content_pos


class ProductDescriptionParser:
    """
    Applies syntactic analysis for practical e-commerce use cases.
    """
    
    def __init__(self):
        self.analyzer = SyntacticAnalyzer()
        self.measurement_patterns = {
            'weight': r'(\d+(?:\.\d+)?)\s*(lbs?|kg|g|oz)',
            'dimensions': r'(\d+(?:\.\d+)?)\s*(in|cm|mm|ft)',
            'price': r'\$\s*(\d+(?:\.\d+)?)'
        }
    
    def parse_description(self, description: str) -> Dict[str, Dict[str, float]]:
        """
        Extracts structured information from product descriptions.
        
        Args:
            description: Product description text
            
        Returns:
            Dictionary with normalized measurements and attributes
        """
        try:
            result = {
                'measurements': {},
                'attributes': {}
            }
            
            # Extract and normalize measurements
            for measure_type, pattern in self.measurement_patterns.items():
                matches = re.finditer(pattern, description, re.IGNORECASE)
                for match in matches:
                    value = float(match.group(1))
                    unit = match.group(2).lower()
                    
                    # Normalize to standard units (kg, cm, USD)
                    if measure_type == 'weight':
                        value = self._normalize_weight(value, unit)
                        result['measurements'][measure_type] = {'value': value, 'unit': 'kg'}
                    elif measure_type == 'dimensions':
                        value = self._normalize_length(value, unit)
                        result['measurements'][measure_type] = {'value': value, 'unit': 'cm'}
                    elif measure_type == 'price':
                        result['measurements'][measure_type] = {'value': value, 'unit': 'USD'}
            
            # Extract other attributes using POS analysis
            pos_analysis = self.analyzer.analyze_pos_tags(description)
            for word_info in pos_analysis:
                if word_info['is_content_word']:
                    result['attributes'][word_info['lemma']] = word_info['word']
            
            return result
            
        except Exception as e:
            print(f"Error in product description parsing: {str(e)}")
            return {'measurements': {}, 'attributes': {}}
    
    def _normalize_weight(self, value: float, unit: str) -> float:
        """
        Normalizes weight to kilograms.
        """
        conversions = {
            'lb': 0.453592,
            'lbs': 0.453592,
            'oz': 0.0283495,
            'g': 0.001,
            'kg': 1.0
        }
        return value * conversions.get(unit, 1.0)
    
    def _normalize_length(self, value: float, unit: str) -> float:
        """
        Normalizes length to centimeters.
        """
        conversions = {
            'in': 2.54,
            'ft': 30.48,
            'mm': 0.1,
            'cm': 1.0
        }
        return value * conversions.get(unit, 1.0)

def main():
    # Initialize analyzers
    syntactic_analyzer = SyntacticAnalyzer()
    product_parser = ProductDescriptionParser()
    
    # Example 1: Sentence Boundary Detection
    print("\n=== Sentence Boundary Detection ===")
    text = """Dr. Smith visited St. Mary's Hospital. The patient, Mr. Johnson, 
    was recovering well. The temp. was 98.6 degrees F. Visit www.hospital.com 
    for more info."""
    
    sentences = syntactic_analyzer.detect_sentence_boundaries(text)
    print("\nDetected sentences:")
    for i, sent in enumerate(sentences, 1):
        print(f"{i}. {sent.strip()}")
    
    # Example 2: POS Analysis
    print("\n=== POS Analysis ===")
    sentence = "The quick brown fox jumps over the lazy dog"
    pos_analysis = syntactic_analyzer.analyze_pos_tags(sentence)
    
    print("\nPOS Analysis:")
    for item in pos_analysis:
        print(f"Word: {item['word']:<10} POS: {item['pos']:<6} "
              f"Lemma: {item['lemma']:<10} "
              f"Content Word: {'Yes' if item['is_content_word'] else 'No'}")
    
    # Example 3: Sentence Structure Analysis
    print("\n=== Sentence Structure Analysis ===")
    sentence = "The waiter cleared the plates"
    structure = syntactic_analyzer.analyze_sentence_structure(sentence)
    
    print("\nStructure Analysis:")
    print("Components:")
    for role, words in structure.items():
        if words:
            print(f"- {role}:")
            for word in words:
                print(f"  - {word}")
    
    # Example 4: Product Description Parsing
    print("\n=== Product Description Parsing ===")
    product_desc = """Acer Aspire E15 Laptop, Model: A515-43-R19L
    15.6" FHD Display, AMD Ryzen 5 3500U
    8GB DDR4 RAM, 256GB SSD
    Weight: 4.19 lbs
    Dimensions: 15.02 x 10.2 x 0.71 inches"""
    
    parsed = product_parser.parse_description(product_desc)
    print("\nExtracted Information:")
    for field, info in parsed['measurements'].items():
        print(f"\n{field.title()}:")
        print(f"- Value: {info['value']}")
        print(f"- Unit: {info['unit']}")
    
    for field, value in parsed['attributes'].items():
        print(f"\n{field.title()}:")
        print(f"- {value}")

if __name__ == "__main__":
    main()