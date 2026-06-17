# Python Libraries for Text Processing

## Overview

Python has an extensive ecosystem of libraries specifically designed for text processing tasks. These libraries range from basic string manipulation tools to advanced natural language processing frameworks.

## Built-in String Methods

Before diving into external libraries, Python's built-in string methods provide essential text processing capabilities:

```python
text = "  Hello, World!  "

# Case transformations
text.upper()        # "  HELLO, WORLD!  "
text.lower()        # "  hello, world!  "
text.capitalize()   # "  hello, world!  "

# Whitespace handling
text.strip()        # "Hello, World!"
text.lstrip()       # "Hello, World!  "
text.rstrip()       # "  Hello, World!"

# Substring operations
text.replace("World", "Python")     # "  Hello, Python!  "
text.find("World")                  # 8 (position of substring)
text.startswith("  Hello")          # True
text.endswith("World!  ")           # True

# Splitting and joining
text.split(",")                     # ["  Hello", " World!  "]
" ".join(["Hello", "World"])        # "Hello World"
```

## Regular Expressions (re module)

The `re` module provides powerful pattern matching and text manipulation capabilities:

```python
import re

# Pattern matching
pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
text = "Contact us at info@example.com or support@test.org"
emails = re.findall(pattern, text)
print(emails)  # ['info@example.com', 'support@test.org']

# Substitution
phone = "Call me at 123-456-7890"
digits_only = re.sub(r'\D', '', phone)
print(digits_only)  # '1234567890'

# Advanced pattern matching with groups
date_text = "Today is 2023-05-15"
match = re.search(r'(\d{4})-(\d{2})-(\d{2})', date_text)
if match:
    year, month, day = match.groups()
    print(f"Year: {year}, Month: {month}, Day: {day}")
```

## Natural Language Toolkit (NLTK)

NLTK is a comprehensive library for natural language processing:

```python
# Installation: pip install nltk
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required data (run once)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

def nltk_text_processing_example(text):
    # Tokenization
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in words if w.lower() not in stop_words and w.isalpha()]
    
    # Stemming
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(w) for w in filtered_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w) for w in filtered_words]
    
    return {
        'sentences': sentences,
        'words': words,
        'filtered': filtered_words,
        'stemmed': stemmed,
        'lemmatized': lemmatized
    }

sample_text = "The quick brown foxes are jumping over the lazy dogs."
processed = nltk_text_processing_example(sample_text)
print(processed)
```

## spaCy

spaCy is an industrial-strength NLP library offering speed and accuracy:

```python
# Installation: pip install spacy
# Download model: python -m spacy download en_core_web_sm
import spacy

# Load English language model
nlp = spacy.load("en_core_web_sm")

def spacy_text_analysis(text):
    doc = nlp(text)
    
    # Extract entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Extract noun phrases
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    
    # Part-of-speech tagging
    pos_tags = [(token.text, token.pos_) for token in doc if not token.is_space]
    
    # Dependency parsing
    dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
    
    return {
        'entities': entities,
        'noun_phrases': noun_phrases,
        'pos_tags': pos_tags,
        'dependencies': dependencies
    }

text = "Apple is looking at buying U.K. startup for $1 billion"
analysis = spacy_text_analysis(text)
print(analysis)
```

## TextBlob

TextBlob provides a simpler interface for common NLP tasks:

```python
# Installation: pip install textblob
from textblob import TextBlob

def textblob_example(text):
    blob = TextBlob(text)
    
    # Sentiment analysis
    sentiment = blob.sentiment
    
    # Parts of speech tagging
    pos_tags = blob.tags
    
    # Noun phrase extraction
    noun_phrases = blob.noun_phrases
    
    # Translation
    translated = blob.translate(to='es')  # Translate to Spanish
    
    # Spelling correction
    corrected = TextBlob("I havv goood speling").correct()
    
    return {
        'polarity': sentiment.polarity,  # -1 to 1, negative to positive
        'subjectivity': sentiment.subjectivity,  # 0 to 1, objective to subjective
        'pos_tags': pos_tags,
        'noun_phrases': list(noun_phrases),
        'translation': str(translated),
        'correction': str(corrected)
    }

text = "TextBlob is amazingly simple to use for common NLP tasks."
results = textblob_example(text)
print(results)
```

## Pandas for Text Processing

Pandas provides efficient text processing capabilities for structured data:

```python
import pandas as pd

# Create sample data
df = pd.DataFrame({
    'text': [
        'Python is great for data science',
        'Natural language processing with Python',
        'Machine learning applications'
    ],
    'label': ['DS', 'NLP', 'ML']
})

# Apply text transformations
df['uppercase'] = df['text'].str.upper()
df['length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['contains_python'] = df['text'].str.contains('Python', case=False)

# Extract using regex
df['first_word'] = df['text'].str.extract(r'^(\w+)')
print(df)
```

## Regular Expression Libraries

### regex (Enhanced Regular Expression Library)

An alternative to the standard `re` module with additional features:

```python
# Installation: pip install regex
import regex as re

# Unicode support
text = "Café naïve résumé"
matches = re.findall(r'\p{L}+', text)  # Matches any Unicode letter
print(matches)  # ['Café', 'naïve', 'résumé']

# Variable-length lookbehind
text = "abc123def456ghi"
matches = re.findall(r'(?<=\d{3})\w+', text)
print(matches)  # ['def', 'ghi']
```

## Hugging Face Transformers

State-of-the-art models for various NLP tasks:

```python
# Installation: pip install transformers torch
from transformers import pipeline

# Sentiment analysis
classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]

# Named Entity Recognition
ner = pipeline("ner", aggregation_strategy="simple")
entities = ner("Hugging Face is located in New York City")
print(entities)

# Text generation
generator = pipeline("text-generation", model="gpt2")
result = generator("The future of AI is", max_length=30, do_sample=True)
print(result[0]['generated_text'])
```

## Selecting the Right Library

Choose libraries based on your specific needs:

- **Basic string operations**: Use Python's built-in methods
- **Pattern matching**: Use the `re` module
- **Academic/NLP research**: NLTK for comprehensive tools
- **Production applications**: spaCy for speed and accuracy
- **Quick prototyping**: TextBlob for ease of use
- **Large datasets**: Pandas for structured text data
- **Deep learning**: Transformers for state-of-the-art models

## Performance Considerations

- For large texts, consider processing in chunks
- Use vectorized operations when available (pandas)
- Cache processed results when possible
- Profile your code to identify bottlenecks
- Consider parallel processing for CPU-intensive tasks