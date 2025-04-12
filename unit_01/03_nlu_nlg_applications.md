# Natural Language Understanding (NLU) and Generation (NLG) Applications

## Introduction
This module explores practical applications of Natural Language Understanding (NLU) and Natural Language Generation (NLG), with implementation examples and best practices.

## 1. Natural Language Understanding (NLU)

### Text Classification Implementation
```python
from typing import List, Dict, Union
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class TextClassifier:
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
    def classify(self, text: str) -> Dict[str, float]:
        """Classify text into predefined categories"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        return {
            label: float(prob)
            for label, prob in zip(self.model.config.id2label.values(), probs[0])
        }

class SentimentAnalyzer:
    def __init__(self):
        self.classifier = TextClassifier("distilbert-base-uncased-finetuned-sst-2-english")
        
    def analyze_sentiment(self, text: str) -> Dict[str, Union[str, float]]:
        """Analyze sentiment of given text"""
        sentiment_scores = self.classifier.classify(text)
        
        return {
            'text': text,
            'sentiment': max(sentiment_scores.items(), key=lambda x: x[1])[0],
            'confidence': max(sentiment_scores.values()),
            'scores': sentiment_scores
        }
```

### Named Entity Recognition
```python
import spacy
from typing import List, Dict, Tuple

class EntityRecognizer:
    def __init__(self, model: str = "en_core_web_sm"):
        self.nlp = spacy.load(model)
        
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities from text"""
        doc = self.nlp(text)
        
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'description': spacy.explain(ent.label_)
            })
        
        return entities
    
    def analyze_entity_relationships(self, text: str) -> List[Dict]:
        """Analyze relationships between entities"""
        doc = self.nlp(text)
        relationships = []
        
        for token in doc:
            if token.dep_ in ['nsubj', 'dobj'] and token.head.pos_ == 'VERB':
                relationships.append({
                    'subject': token.text,
                    'verb': token.head.text,
                    'relationship_type': token.dep_
                })
        
        return relationships
```

### Intent Recognition System
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

class IntentRecognizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.classifier = MultinomialNB()
        self.intents = []
        
    def train(self, training_data: List[Dict[str, Union[str, str]]]):
        """Train the intent recognizer"""
        texts = [item['text'] for item in training_data]
        self.intents = list(set(item['intent'] for item in training_data))
        
        X = self.vectorizer.fit_transform(texts)
        y = [self.intents.index(item['intent']) for item in training_data]
        
        self.classifier.fit(X, y)
    
    def recognize_intent(self, text: str) -> Dict[str, Union[str, float]]:
        """Recognize intent from text"""
        X = self.vectorizer.transform([text])
        probs = self.classifier.predict_proba(X)[0]
        
        return {
            'text': text,
            'intent': self.intents[np.argmax(probs)],
            'confidence': float(max(probs)),
            'all_intents': {
                intent: float(prob)
                for intent, prob in zip(self.intents, probs)
            }
        }
```

## 2. Natural Language Generation (NLG)

### Text Generation System
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class TextGenerator:
    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        num_return_sequences: int = 1,
        temperature: float = 0.7
    ) -> List[str]:
        """Generate text based on prompt"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            no_repeat_ngram_size=2,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
```

### Template-Based Generation
```python
from string import Template
from typing import Dict, List, Union
import re

class TemplateGenerator:
    def __init__(self):
        self.templates = {}
        
    def add_template(self, name: str, template: str):
        """Add a new template"""
        self.templates[name] = Template(template)
        
    def generate(self, template_name: str, data: Dict[str, str]) -> str:
        """Generate text using template"""
        if template_name not in self.templates:
            raise KeyError(f"Template '{template_name}' not found")
            
        return self.templates[template_name].safe_substitute(data)
    
    def batch_generate(
        self,
        template_name: str,
        data_list: List[Dict[str, str]]
    ) -> List[str]:
        """Generate multiple texts using the same template"""
        return [self.generate(template_name, data) for data in data_list]
```

### Summarization System
```python
from transformers import pipeline
from typing import Dict, Union, List

class TextSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.summarizer = pipeline("summarization", model=model_name)
        
    def summarize(
        self,
        text: str,
        max_length: int = 130,
        min_length: int = 30,
        do_sample: bool = False
    ) -> Dict[str, Union[str, int]]:
        """Generate summary of input text"""
        summary = self.summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample
        )[0]
        
        return {
            'original_text': text,
            'summary': summary['summary_text'],
            'original_length': len(text.split()),
            'summary_length': len(summary['summary_text'].split())
        }
    
    def batch_summarize(
        self,
        texts: List[str],
        **kwargs
    ) -> List[Dict[str, Union[str, int]]]:
        """Summarize multiple texts"""
        return [self.summarize(text, **kwargs) for text in texts]
```

## 3. Practical Applications

### 1. Chatbot Implementation
```python
class Chatbot:
    def __init__(self):
        self.intent_recognizer = IntentRecognizer()
        self.entity_recognizer = EntityRecognizer()
        self.text_generator = TextGenerator()
        self.context = {}
        
    def process_message(self, message: str) -> Dict:
        """Process user message and generate response"""
        # Understand intent
        intent = self.intent_recognizer.recognize_intent(message)
        
        # Extract entities
        entities = self.entity_recognizer.extract_entities(message)
        
        # Update context
        self.update_context(intent, entities)
        
        # Generate response
        response = self.generate_response(intent, entities)
        
        return {
            'user_message': message,
            'intent': intent,
            'entities': entities,
            'response': response,
            'context': self.context
        }
    
    def update_context(self, intent: Dict, entities: List[Dict]):
        """Update conversation context"""
        self.context.update({
            'last_intent': intent['intent'],
            'entities': entities
        })
    
    def generate_response(self, intent: Dict, entities: List[Dict]) -> str:
        """Generate appropriate response based on intent and entities"""
        prompt = f"Respond to a user with intent {intent['intent']}"
        if entities:
            prompt += f" mentioning {', '.join(e['text'] for e in entities)}"
        
        responses = self.text_generator.generate_text(prompt)
        return responses[0]
```

### 2. Document Analysis System
```python
class DocumentAnalyzer:
    def __init__(self):
        self.entity_recognizer = EntityRecognizer()
        self.summarizer = TextSummarizer()
        self.sentiment_analyzer = SentimentAnalyzer()
        
    def analyze_document(self, text: str) -> Dict:
        """Perform comprehensive document analysis"""
        # Get entities
        entities = self.entity_recognizer.extract_entities(text)
        
        # Generate summary
        summary = self.summarizer.summarize(text)
        
        # Analyze sentiment
        sentiment = self.sentiment_analyzer.analyze_sentiment(text)
        
        # Analyze entity relationships
        relationships = self.entity_recognizer.analyze_entity_relationships(text)
        
        return {
            'summary': summary['summary'],
            'entities': entities,
            'sentiment': sentiment,
            'relationships': relationships,
            'original_text': text
        }
```

## Best Practices

### 1. NLU Best Practices
- Preprocess text thoroughly
- Handle edge cases and errors gracefully
- Implement confidence thresholds
- Maintain context when needed
- Regular model updates and retraining

### 2. NLG Best Practices
- Implement output validation
- Control generation parameters
- Handle toxic output
- Maintain consistency
- Implement fallback mechanisms

## Common Challenges

### 1. NLU Challenges
- Ambiguity resolution
- Context understanding
- Handling colloquialisms
- Multiple languages
- Domain adaptation

### 2. NLG Challenges
- Maintaining coherence
- Ensuring factual accuracy
- Style consistency
- Avoiding bias
- Performance optimization

## References
1. Jurafsky, D. & Martin, J.H. "Speech and Language Processing"
2. Manning, C.D. et al. "Introduction to Information Retrieval"
3. Vaswani, A. et al. "Attention Is All You Need"

---
*Note: The implementations provided are for educational purposes. Production use may require additional error handling, optimization, and security measures.* 