"""
Practical comparison of shallow and deep NLP approaches.
This module demonstrates the differences in capabilities and performance
between shallow and deep NLP techniques.
"""

import re
import time
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
from transformers import pipeline
import spacy
import numpy as np

class ShallowNLP:
    """Implements shallow NLP approaches using rule-based and statistical methods."""
    
    def __init__(self):
        """Initialize the ShallowNLP class with spaCy model and sentiment lexicons."""
        self.nlp = spacy.load("en_core_web_sm")
        self.positive_words = set(['good', 'great', 'excellent', 'amazing', 'innovative', 'efficient'])
        self.negative_words = set(['bad', 'poor', 'terrible', 'inefficient', 'problematic'])
        
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract entities using spaCy's statistical model."""
        doc = self.nlp(text)
        return [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using a rule-based approach with word lexicons."""
        doc = self.nlp(text.lower())
        words = [token.text for token in doc if not token.is_stop and not token.is_punct]
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        score = (positive_count - negative_count) / (positive_count + negative_count + 1)
        sentiment = 'positive' if score > 0 else 'negative' if score < 0 else 'neutral'
        
        return {
            'sentiment': sentiment,
            'score': score,
            'positive_words': positive_count,
            'negative_words': negative_count
        }
    
    def summarize_text(self, text: str, num_sentences: int = 3) -> str:
        """Create a basic extractive summary using sentence importance scoring."""
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        # Simple scoring based on sentence position and length
        scores = []
        for i, sent in enumerate(sentences):
            position_score = 1.0 / (i + 1)  # Earlier sentences get higher scores
            length_score = min(len(sent.split()) / 20.0, 1.0)  # Favor medium-length sentences
            scores.append(position_score * length_score)
        
        # Get top sentences
        top_indices = np.argsort(scores)[-num_sentences:]
        summary = ' '.join([sentences[i] for i in sorted(top_indices)])
        return summary

class DeepNLP:
    """Implements deep learning-based NLP approaches using transformers."""
    
    def __init__(self):
        """Initialize the DeepNLP class with transformer pipelines."""
        self.ner_pipeline = pipeline("ner")
        self.sentiment_pipeline = pipeline("sentiment-analysis")
        self.summarizer = pipeline("summarization")
        self.qa_pipeline = pipeline("question-answering")
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using transformer-based NER."""
        return self.ner_pipeline(text)
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using a pre-trained transformer."""
        result = self.sentiment_pipeline(text)[0]
        return {
            'sentiment': result['label'],
            'score': result['score']
        }
    
    def summarize_text(self, text: str, max_length: int = 130) -> str:
        """Generate an abstractive summary using transformers."""
        return self.summarizer(text, max_length=max_length, min_length=30)[0]['summary_text']
    
    def answer_question(self, context: str, question: str) -> Dict[str, Any]:
        """Answer questions about the given context using transformers."""
        return self.qa_pipeline(question=question, context=context)

def compare_approaches(text: str, question: Optional[str] = None) -> Dict[str, Any]:
    """Compare shallow and deep NLP approaches on the same text."""
    shallow_nlp = ShallowNLP()
    deep_nlp = DeepNLP()
    
    results = {
        'shallow_nlp': {
            'entities': shallow_nlp.extract_entities(text),
            'sentiment': shallow_nlp.analyze_sentiment(text),
            'summary': shallow_nlp.summarize_text(text)
        },
        'deep_nlp': {
            'entities': deep_nlp.extract_entities(text),
            'sentiment': deep_nlp.analyze_sentiment(text),
            'summary': deep_nlp.summarize_text(text)
        }
    }
    
    if question:
        results['deep_nlp']['qa_result'] = deep_nlp.answer_question(text, question)
    
    return results

if __name__ == "__main__":
    # Example usage
    sample_text = """
    Google has announced a groundbreaking new AI model that revolutionizes natural language processing. 
    The innovative system demonstrates remarkable efficiency in understanding and generating human-like text, 
    setting new benchmarks in various NLP tasks. However, some experts express concerns about the model's 
    computational requirements and environmental impact. The company plans to release a more efficient version 
    in the coming months, addressing these challenges while maintaining high performance.
    """
    
    sample_question = "What are the concerns about the new AI model?"
    
    results = compare_approaches(sample_text, sample_question)
    
    print("=== Shallow NLP Results ===")
    print("Entities:", results['shallow_nlp']['entities'])
    print("Sentiment:", results['shallow_nlp']['sentiment'])
    print("Summary:", results['shallow_nlp']['summary'])
    
    print("\n=== Deep NLP Results ===")
    print("Entities:", results['deep_nlp']['entities'])
    print("Sentiment:", results['deep_nlp']['sentiment'])
    print("Summary:", results['deep_nlp']['summary'])
    print("Answer:", results['deep_nlp']['qa_result']) 