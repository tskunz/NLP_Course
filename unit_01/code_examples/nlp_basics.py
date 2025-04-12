"""
Basic NLP examples demonstrating core concepts from Unit 1
"""

import spacy
from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

def demonstrate_natural_language_complexity():
    """
    Demonstrates the complexity of natural language processing
    by showing various challenges in text analysis
    """
    # Example 1: Ambiguity in natural language
    text = "I saw a man on a hill with a telescope"
    print("Example of ambiguity:")
    print("Text:", text)
    print("Possible interpretations:")
    print("1. I used a telescope to see a man on a hill")
    print("2. I saw a man who was on a hill and had a telescope")
    print("3. I saw a man on a hill that had a telescope mounted on it")
    print()

    # Example 2: Context dependency
    nlp = spacy.load("en_core_web_sm")
    sentences = [
        "The bank is closed today",
        "The bank of the river is muddy"
    ]
    print("Example of context dependency:")
    for sentence in sentences:
        doc = nlp(sentence)
        print(f"Sentence: {sentence}")
        print(f"'bank' meaning depends on context - Entities found:", 
              [(ent.text, ent.label_) for ent in doc.ents])
    print()

def demonstrate_artificial_language_properties():
    """
    Shows properties of artificial languages through programming examples
    """
    # Example 1: Unambiguous syntax
    code_example = '''
    def calculate_area(length: float, width: float) -> float:
        """Calculate rectangle area"""
        return length * width
    '''
    print("Example of unambiguous syntax in programming languages:")
    print(code_example)
    print()

    # Example 2: Context-free interpretation
    expression = "3 + 4 * 2"
    print("Example of context-free interpretation:")
    print(f"Expression: {expression}")
    print(f"Result will always be {3 + 4 * 2} due to fixed operator precedence")
    print()

def demonstrate_nlu_example():
    """
    Demonstrates Natural Language Understanding with sentiment analysis
    """
    # Using transformers for sentiment analysis
    classifier = pipeline("sentiment-analysis")
    
    texts = [
        "This product is amazing! I love it!",
        "The service was terrible and I want my money back.",
        "The movie was okay, nothing special."
    ]
    
    print("Natural Language Understanding Example - Sentiment Analysis:")
    for text in texts:
        result = classifier(text)[0]
        print(f"\nText: {text}")
        print(f"Sentiment: {result['label']}")
        print(f"Confidence: {result['score']:.2f}")
    print()

def demonstrate_nlg_example():
    """
    Demonstrates Natural Language Generation with text completion
    """
    # Using transformers for text generation
    generator = pipeline("text-generation", model="gpt2")
    
    prompts = [
        "The future of artificial intelligence",
        "Once upon a time in a digital"
    ]
    
    print("Natural Language Generation Example - Text Completion:")
    for prompt in prompts:
        result = generator(prompt, max_length=50, num_return_sequences=1)[0]
        print(f"\nPrompt: {prompt}")
        print(f"Generated text: {result['generated_text']}")
    print()

if __name__ == "__main__":
    print("=== Unit 1: Introduction to NLP ===\n")
    demonstrate_natural_language_complexity()
    demonstrate_artificial_language_properties()
    demonstrate_nlu_example()
    demonstrate_nlg_example() 