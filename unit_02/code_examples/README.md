# Syntactic Analysis Examples

This directory contains practical examples of syntactic analysis using Python and NLTK.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download required NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
```

## Usage

The `syntactic_analysis.py` file contains two main classes:

1. `SyntacticAnalyzer`: For general syntactic analysis tasks
   - Sentence boundary detection
   - POS tagging with lemmatization
   - Basic sentence structure analysis

2. `ProductDescriptionParser`: For practical application in e-commerce
   - Structured information extraction
   - Measurement normalization
   - Product attribute parsing

## Example Usage

```python
from syntactic_analysis import SyntacticAnalyzer, ProductDescriptionParser

# Create analyzer instance
analyzer = SyntacticAnalyzer()

# Detect sentence boundaries
text = "Mr. Smith bought 3.5 lbs. of apples. He paid $5.99."
sentences = analyzer.detect_sentence_boundaries(text)

# Analyze POS tags
analysis = analyzer.analyze_pos_tags("The quick brown fox jumps.")

# Parse product descriptions
parser = ProductDescriptionParser()
description = "Model X100: 15.6 in. laptop, weight: 4.2 lbs, Price: $999.99"
parsed_info = parser.parse_description(description)
```

## Notes

- The code includes extensive error handling and edge cases
- Measurement normalization supports common units
- POS tagging includes content word identification 