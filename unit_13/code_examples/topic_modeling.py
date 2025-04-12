"""
Topic Modeling Implementation Examples
This module provides implementations of various topic modeling approaches,
including LDA, NMF, and entity-centric topic modeling.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from gensim.models import LdaModel, CoherenceModel
from gensim.corpora import Dictionary
import spacy
from collections import defaultdict
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

class TopicModelPreprocessor:
    """Handles text preprocessing for topic modeling."""
    
    def __init__(self, language: str = 'en'):
        """Initialize the preprocessor with specified language model."""
        self.nlp = spacy.load(language)
        self.stop_words = self.nlp.Defaults.stop_words
    
    def preprocess_documents(self, documents: List[str]) -> List[List[str]]:
        """
        Preprocess documents for topic modeling.
        
        Args:
            documents: List of document strings
            
        Returns:
            List of tokenized and preprocessed documents
        """
        processed_docs = []
        for doc in documents:
            # Process with spaCy
            tokens = self.nlp(doc.lower())
            
            # Filter tokens
            filtered_tokens = [
                token.lemma_ for token in tokens
                if (not token.is_stop and 
                    not token.is_punct and
                    not token.is_space and
                    len(token.text) > 2)
            ]
            
            processed_docs.append(filtered_tokens)
        
        return processed_docs

class LDATopicModel:
    """Implementation of LDA topic modeling."""
    
    def __init__(self, 
                 num_topics: int = 10,
                 passes: int = 10,
                 alpha: str = 'auto'):
        """
        Initialize LDA model with parameters.
        
        Args:
            num_topics: Number of topics to extract
            passes: Number of passes through corpus
            alpha: Prior document-topic distribution
        """
        self.num_topics = num_topics
        self.passes = passes
        self.alpha = alpha
        self.model = None
        self.dictionary = None
        self.corpus = None
    
    def fit(self, processed_documents: List[List[str]]) -> None:
        """
        Train the LDA model on processed documents.
        
        Args:
            processed_documents: List of preprocessed and tokenized documents
        """
        # Create dictionary and corpus
        self.dictionary = Dictionary(processed_documents)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in processed_documents]
        
        # Train LDA model
        self.model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            random_state=42,
            passes=self.passes,
            alpha=self.alpha,
            per_word_topics=True
        )
    
    def get_topics(self, num_words: int = 10) -> List[List[Tuple[str, float]]]:
        """Get the top words for each topic."""
        return [self.model.show_topic(topic_id, num_words)
                for topic_id in range(self.num_topics)]
    
    def get_document_topics(self, doc: List[str]) -> List[Tuple[int, float]]:
        """Get topic distribution for a document."""
        bow = self.dictionary.doc2bow(doc)
        return self.model.get_document_topics(bow)
    
    def visualize(self, output_path: str) -> None:
        """Create and save interactive visualization."""
        vis_data = pyLDAvis.gensim_models.prepare(
            self.model, self.corpus, self.dictionary
        )
        pyLDAvis.save_html(vis_data, output_path)

class NMFTopicModel:
    """Implementation of NMF topic modeling."""
    
    def __init__(self, 
                 num_topics: int = 10,
                 max_features: int = 5000):
        """
        Initialize NMF model with parameters.
        
        Args:
            num_topics: Number of topics to extract
            max_features: Maximum number of features for TF-IDF
        """
        self.num_topics = num_topics
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.model = NMF(n_components=num_topics, random_state=42)
        self.feature_names = None
        self.document_topics = None
    
    def fit(self, documents: List[str]) -> None:
        """
        Train the NMF model on documents.
        
        Args:
            documents: List of document strings
        """
        # Create document-term matrix
        dtm = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Apply NMF
        self.document_topics = self.model.fit_transform(dtm)
    
    def get_topics(self, num_words: int = 10) -> List[List[Tuple[str, float]]]:
        """Get the top words for each topic."""
        topics = []
        for topic_idx, topic in enumerate(self.model.components_):
            top_words = [
                (self.feature_names[i], score)
                for i, score in sorted(
                    enumerate(topic),
                    key=lambda x: x[1],
                    reverse=True
                )[:num_words]
            ]
            topics.append(top_words)
        return topics
    
    def get_document_topics(self, doc: str) -> np.ndarray:
        """Get topic distribution for a document."""
        doc_vector = self.vectorizer.transform([doc])
        return self.model.transform(doc_vector)[0]
    
    def visualize_topics(self, output_path: str) -> None:
        """Create and save topic visualization."""
        # Create word clouds for each topic
        for idx, topic in enumerate(self.get_topics(num_words=50)):
            wordcloud = WordCloud(
                background_color='white',
                width=800,
                height=400
            ).generate_from_frequencies(dict(topic))
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Topic {idx + 1}')
            plt.savefig(f'{output_path}_topic_{idx + 1}.png')
            plt.close()

class EntityCentricTopicModel:
    """Implementation of entity-centric topic modeling."""
    
    def __init__(self, 
                 num_topics: int = 10,
                 language: str = 'en'):
        """
        Initialize entity-centric topic model.
        
        Args:
            num_topics: Number of topics to extract
            language: Language model to use
        """
        self.num_topics = num_topics
        self.nlp = spacy.load(language)
        self.entity_topics = defaultdict(list)
        self.topic_entities = defaultdict(list)
    
    def extract_entities(self, doc: str) -> List[Tuple[str, str]]:
        """Extract named entities from document."""
        processed_doc = self.nlp(doc)
        return [(ent.text, ent.label_) for ent in processed_doc.ents]
    
    def fit(self, documents: List[str], base_model: LDATopicModel) -> None:
        """
        Train the entity-centric model using a base topic model.
        
        Args:
            documents: List of document strings
            base_model: Trained LDA model
        """
        for doc_idx, doc in enumerate(documents):
            # Get document topics
            doc_topics = base_model.get_document_topics(
                base_model.dictionary.doc2bow(doc.split())
            )
            
            # Extract entities
            entities = self.extract_entities(doc)
            
            # Associate entities with topics
            for entity, entity_type in entities:
                for topic_id, topic_prob in doc_topics:
                    self.entity_topics[entity].append((topic_id, topic_prob))
                    self.topic_entities[topic_id].append((entity, entity_type))
    
    def get_entity_topics(self, entity: str) -> List[Tuple[int, float]]:
        """Get topics associated with an entity."""
        if entity not in self.entity_topics:
            return []
        
        # Average topic probabilities for the entity
        topic_probs = defaultdict(list)
        for topic_id, prob in self.entity_topics[entity]:
            topic_probs[topic_id].append(prob)
        
        return [(topic_id, np.mean(probs))
                for topic_id, probs in topic_probs.items()]
    
    def get_topic_entities(self, topic_id: int,
                          entity_type: Optional[str] = None) -> List[Tuple[str, float]]:
        """Get entities associated with a topic."""
        if topic_id not in self.topic_entities:
            return []
        
        entities = self.topic_entities[topic_id]
        if entity_type:
            entities = [e for e in entities if e[1] == entity_type]
        
        # Count entity occurrences
        entity_counts = defaultdict(int)
        for entity, _ in entities:
            entity_counts[entity] += 1
        
        # Normalize counts
        total = sum(entity_counts.values())
        return [(entity, count/total)
                for entity, count in entity_counts.items()]

def main():
    """Example usage of topic modeling implementations."""
    # Sample documents
    documents = [
        "Machine learning algorithms are transforming artificial intelligence research.",
        "Deep neural networks achieve state-of-the-art results in computer vision tasks.",
        "Natural language processing helps computers understand human language.",
        "Reinforcement learning enables agents to learn from their environment.",
        "Data science combines statistics and programming to analyze big data."
    ]
    
    # Initialize preprocessor
    preprocessor = TopicModelPreprocessor()
    processed_docs = preprocessor.preprocess_documents(documents)
    
    # LDA Topic Modeling
    print("\nLDA Topic Modeling Example:")
    lda_model = LDATopicModel(num_topics=3)
    lda_model.fit(processed_docs)
    
    print("\nLDA Topics:")
    for idx, topic in enumerate(lda_model.get_topics()):
        print(f"\nTopic {idx + 1}:")
        for word, prob in topic:
            print(f"  {word}: {prob:.4f}")
    
    # NMF Topic Modeling
    print("\nNMF Topic Modeling Example:")
    nmf_model = NMFTopicModel(num_topics=3)
    nmf_model.fit(documents)
    
    print("\nNMF Topics:")
    for idx, topic in enumerate(nmf_model.get_topics()):
        print(f"\nTopic {idx + 1}:")
        for word, score in topic:
            print(f"  {word}: {score:.4f}")
    
    # Entity-Centric Topic Modeling
    print("\nEntity-Centric Topic Modeling Example:")
    entity_model = EntityCentricTopicModel(num_topics=3)
    entity_model.fit(documents, lda_model)
    
    # Example entity analysis
    entities = entity_model.extract_entities(documents[0])
    print("\nExtracted Entities:")
    for entity, entity_type in entities:
        print(f"  {entity} ({entity_type})")
        topics = entity_model.get_entity_topics(entity)
        for topic_id, prob in topics:
            print(f"    Topic {topic_id}: {prob:.4f}")

if __name__ == "__main__":
    main() 