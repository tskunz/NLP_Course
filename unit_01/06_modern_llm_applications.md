# Modern LLM Applications and Implementation Examples

## Overview
Large Language Models (LLMs) have revolutionized Natural Language Processing, enabling new applications and improving existing ones. This document explores practical implementations and use cases of LLMs in various domains.

## 1. Advanced Text Generation

### Creative Writing Assistant
```python
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer

class CreativeWritingAssistant:
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained('gpt2-large')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
        self.generator = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer)
    
    def generate_story_continuation(self, prompt: str, max_length: int = 200):
        """Generate creative continuation of a story prompt"""
        try:
            response = self.generator(
                prompt,
                max_length=max_length,
                num_return_sequences=3,
                temperature=0.9,
                top_k=50,
                top_p=0.95,
                do_sample=True
            )
            return [r['generated_text'] for r in response]
        except Exception as e:
            return f"Error generating story: {str(e)}"
    
    def generate_with_style(self, content: str, style: str):
        """Generate text in a specific style"""
        style_prompt = f"Write the following in a {style} style: {content}"
        return self.generate_story_continuation(style_prompt, max_length=150)[0]

# Example usage
assistant = CreativeWritingAssistant()
story_start = "The old lighthouse stood abandoned on the rocky coast..."
continuations = assistant.generate_story_continuation(story_start)
noir_style = assistant.generate_with_style("The detective entered the room", "film noir")
```

### Code Generation and Documentation
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class CodeAssistant:
    def __init__(self, model_name="Salesforce/codegen-350M-mono"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
    def generate_code(self, prompt: str, max_length: int = 200):
        """Generate code based on natural language description"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def generate_documentation(self, code: str):
        """Generate documentation for provided code"""
        prompt = f"Write detailed documentation for this code:\n{code}\n\nDocumentation:"
        return self.generate_code(prompt)
    
    def suggest_improvements(self, code: str):
        """Suggest code improvements"""
        prompt = f"Suggest improvements for this code:\n{code}\n\nImprovements:"
        return self.generate_code(prompt)

# Example usage
assistant = CodeAssistant()
code_prompt = "Write a Python function to calculate the Fibonacci sequence"
generated_code = assistant.generate_code(code_prompt)
documentation = assistant.generate_documentation(generated_code)
```

## 2. Intelligent Conversation Systems

### Context-Aware Chatbot
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

class ContextAwareChat:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.conversation_history = []
        
    def maintain_context(self, max_history: int = 5):
        """Maintain recent conversation context"""
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]
    
    def generate_response(self, user_input: str):
        """Generate contextual response"""
        # Add user input to history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Create context from history
        context = " ".join([f"{msg['role']}: {msg['content']}" 
                          for msg in self.conversation_history])
        
        # Generate response
        inputs = self.tokenizer.encode(context + "\nassistant:", 
                                     return_tensors="pt")
        
        outputs = self.model.generate(
            inputs,
            max_length=150,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Add response to history
        self.conversation_history.append({"role": "assistant", "content": response})
        self.maintain_context()
        
        return response

# Example usage
chatbot = ContextAwareChat()
response1 = chatbot.generate_response("Tell me about machine learning.")
response2 = chatbot.generate_response("What are its applications?")
```

## 3. Advanced Document Analysis

### Intelligent Document Processor
```python
from transformers import pipeline
from typing import List, Dict
import spacy

class DocumentProcessor:
    def __init__(self):
        self.summarizer = pipeline("summarization")
        self.qa_model = pipeline("question-answering")
        self.nlp = spacy.load("en_core_web_sm")
        self.classifier = pipeline("zero-shot-classification")
        
    def analyze_document(self, text: str) -> Dict:
        """Comprehensive document analysis"""
        # Generate summary
        summary = self.summarizer(text, max_length=130, min_length=30)[0]['summary_text']
        
        # Extract key information
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        key_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # Classify document
        labels = ["technical", "business", "academic", "news"]
        classification = self.classifier(text, labels)
        
        return {
            "summary": summary,
            "entities": entities,
            "key_phrases": key_phrases,
            "classification": {
                "label": classification["labels"][0],
                "confidence": classification["scores"][0]
            }
        }
    
    def answer_questions(self, document: str, questions: List[str]) -> List[Dict]:
        """Answer multiple questions about the document"""
        answers = []
        for question in questions:
            result = self.qa_model(
                question=question,
                context=document
            )
            answers.append({
                "question": question,
                "answer": result["answer"],
                "confidence": result["score"]
            })
        return answers

# Example usage
processor = DocumentProcessor()
document = """
Machine learning is a subset of artificial intelligence that focuses on developing
systems that can learn from and make decisions based on data. It has numerous
applications in various fields, including healthcare, finance, and technology.
"""

analysis = processor.analyze_document(document)
questions = [
    "What is machine learning?",
    "What are its applications?"
]
answers = processor.answer_questions(document, questions)
```

## 4. Multimodal Applications

### Image-Text Interaction
```python
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import requests

class MultimodalAssistant:
    def __init__(self):
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        self.image_processor = ViTImageProcessor.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        
    def generate_image_caption(self, image_path: str) -> str:
        """Generate natural language caption for an image"""
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert(mode="RGB")
                
            pixel_values = self.image_processor(
                image, return_tensors="pt"
            ).pixel_values
            
            output_ids = self.model.generate(
                pixel_values,
                max_length=50,
                num_beams=4,
                no_repeat_ngram_size=3
            )
            
            caption = self.tokenizer.decode(
                output_ids[0], 
                skip_special_tokens=True
            )
            
            return caption
            
        except Exception as e:
            return f"Error processing image: {str(e)}"
    
    def describe_image_with_context(self, 
                                  image_path: str, 
                                  context: str) -> str:
        """Generate contextual description of an image"""
        base_caption = self.generate_image_caption(image_path)
        
        # Use GPT model to enhance caption with context
        prompt = f"""
        Image caption: {base_caption}
        Context: {context}
        Generate a detailed description incorporating both the image content and context:
        """
        
        # Here you would typically use an LLM to generate the enhanced description
        # For this example, we'll return a formatted version of what we have
        return f"Based on the image showing {base_caption}, and considering the context that {context}, ..."

# Example usage
assistant = MultimodalAssistant()
caption = assistant.generate_image_caption("example_image.jpg")
contextual_description = assistant.describe_image_with_context(
    "example_image.jpg",
    "This image was taken during a tech conference"
)
```

## 5. Advanced Language Understanding

### Semantic Analysis System
```python
from transformers import pipeline
import torch
from typing import List, Dict

class SemanticAnalyzer:
    def __init__(self):
        self.entailment = pipeline("zero-shot-classification")
        self.sentiment = pipeline("sentiment-analysis")
        self.similarity = pipeline(
            "feature-extraction",
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    def analyze_semantic_similarity(self, 
                                 text1: str, 
                                 text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        # Get embeddings
        emb1 = torch.tensor(self.similarity(text1))
        emb2 = torch.tensor(self.similarity(text2))
        
        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            emb1, emb2
        ).item()
        
        return similarity
    
    def analyze_implications(self, 
                           premise: str, 
                           hypotheses: List[str]) -> List[Dict]:
        """Analyze logical implications of a statement"""
        results = []
        for hypothesis in hypotheses:
            classification = self.entailment(
                premise,
                candidate_labels=["entailment", "contradiction", "neutral"]
            )
            results.append({
                "hypothesis": hypothesis,
                "relation": classification["labels"][0],
                "confidence": classification["scores"][0]
            })
        return results
    
    def deep_sentiment_analysis(self, text: str) -> Dict:
        """Perform detailed sentiment analysis"""
        # Basic sentiment
        sentiment = self.sentiment(text)[0]
        
        # Analyze specific aspects
        aspects = ["positive", "negative", "neutral"]
        aspect_scores = self.entailment(
            text,
            candidate_labels=aspects
        )
        
        return {
            "overall_sentiment": sentiment["label"],
            "confidence": sentiment["score"],
            "aspect_analysis": {
                label: score for label, score in zip(
                    aspect_scores["labels"],
                    aspect_scores["scores"]
                )
            }
        }

# Example usage
analyzer = SemanticAnalyzer()
similarity = analyzer.analyze_semantic_similarity(
    "The weather is beautiful today",
    "It's a lovely sunny day"
)
implications = analyzer.analyze_implications(
    "The company reported record profits",
    ["The business is successful", "The economy is growing"]
)
sentiment = analyzer.deep_sentiment_analysis(
    "The product exceeded my expectations but had a few minor issues"
)
```

## Best Practices for LLM Applications

1. **Prompt Engineering**
   - Be specific and clear in instructions
   - Provide relevant context
   - Use consistent formatting
   - Include examples when needed

2. **Error Handling**
   - Implement robust error handling
   - Set appropriate timeouts
   - Handle rate limiting
   - Validate outputs

3. **Performance Optimization**
   - Use appropriate model sizes
   - Implement caching
   - Batch process when possible
   - Monitor resource usage

4. **Ethical Considerations**
   - Implement content filtering
   - Maintain transparency
   - Respect privacy
   - Consider bias mitigation

## References
1. Brown, T., et al. "Language Models are Few-Shot Learners"
2. Raffel, C., et al. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
3. Vaswani, A., et al. "Attention Is All You Need"

---
*Note: The code examples provided are for illustration purposes and may need additional error handling and optimization for production use. Always ensure you have the necessary permissions and licenses when using pre-trained models.* 