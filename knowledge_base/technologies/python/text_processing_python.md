# Text Processing with Python

## Introduction

Text processing refers to the manipulation and analysis of textual data using computational techniques. Python is particularly well-suited for text processing tasks due to its rich standard library and numerous third-party packages designed for text manipulation.

## String Methods

Python strings come with many built-in methods for common text processing tasks:

### Case Conversion
- `upper()` - Convert to uppercase
- `lower()` - Convert to lowercase
- `title()` - Convert to title case
- `capitalize()` - Capitalize first letter

### Whitespace Management
- `strip()` - Remove leading/trailing whitespace
- `lstrip()` - Remove leading whitespace
- `rstrip()` - Remove trailing whitespace

### Search and Replace
- `find(substring)` - Find position of substring
- `replace(old, new)` - Replace occurrences of old with new
- `startswith(prefix)` - Check if string starts with prefix
- `endswith(suffix)` - Check if string ends with suffix

### Splitting and Joining
- `split(separator)` - Split string into list
- `join(iterable)` - Join elements with separator

## Regular Expressions

Regular expressions (regex) allow for powerful pattern matching and text extraction. Import the `re` module:

```python
import re

# Find all email addresses
text = "Contact us at info@example.com or support@company.org"
emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
print(emails)  # ['info@example.com', 'support@company.org']

# Pattern substitution
new_text = re.sub(r'dog', 'cat', 'The dog chased the dog')
print(new_text)  # 'The cat chased the cat'
```

## Text Cleaning

Common text cleaning operations include:

### Removing Special Characters
```python
import re

def clean_text(text):
    # Remove non-alphanumeric characters (keeping spaces)
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Normalize whitespace
    cleaned = ' '.join(cleaned.split())
    return cleaned

raw_text = "  Hello,    world!!! How are you???  "
cleaned = clean_text(raw_text)
print(cleaned)  # "Hello world How are you"
```

### Tokenization
Tokenization splits text into meaningful units (words, sentences):

```python
def simple_tokenize(text):
    # Basic word tokenization
    words = text.split()
    return [word.strip('.,!?";') for word in words if word.strip('.,!?";')]

sentence = "Hello, world! How are you?"
tokens = simple_tokenize(sentence)
print(tokens)  # ['Hello', 'world', 'How', 'are', 'you']
```

## File Text Processing

Reading and processing text from files:

```python
def count_words_in_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
        
    # Count words
    words = content.split()
    return len(words)

def process_large_file(filename):
    # Process large files line by line to conserve memory
    word_count = 0
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            word_count += len(line.split())
    return word_count
```

## Advanced Text Processing with NLTK

NLTK (Natural Language Toolkit) provides advanced text processing capabilities:

```python
# Install with: pip install nltk
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required data (run once)
# nltk.download('punkt')
# nltk.download('stopwords')

def advanced_tokenize(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in words if w.lower() not in stop_words and w.isalpha()]
    
    # Stemming
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(w) for w in filtered_words]
    
    return sentences, filtered_words, stemmed
```

## Using Pandas for Text Processing

For larger datasets, pandas can help manage and process text efficiently:

```python
import pandas as pd

# Create a DataFrame with text data
df = pd.DataFrame({
    'text': [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?'
    ]
})

# Apply text transformations
df['uppercase'] = df['text'].str.upper()
df['length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()

print(df.head())
```

## Conclusion

Python offers numerous tools for text processing, from simple string methods to sophisticated NLP libraries. The key is selecting the right approach based on your specific needs and the complexity of the text processing task.