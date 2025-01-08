# Working with Text Data

## 2.1 Understanding Word Embeddings

### Why Word Embeddings Are Needed

- Deep neural networks, including LLMs, cannot process raw text directly.
- Text is categorical and incompatible with mathematical operations in neural networks.
- Words are represented as continuous-valued vectors for neural networks to process.
- **Embedding**: The process of converting data into a vector format using specific layers or pretrained models.

### Embedding Models for Various Data Formats

- Different embedding models are required for different data types (e.g., text, video, audio).
- Embedding maps discrete objects (e.g., words, images) to points in a continuous vector space.
- Embeddings are used to make non-numeric data compatible with neural networks.

### Word Embeddings in Context

- **Word embeddings**: Map individual words to continuous vector representations.
- **Sentence/paragraph embeddings**: Extend the concept to longer text structures, often used in retrieval-augmented generation.
- The focus in this chapter is on word embeddings for training GPT-like LLMs.

### Popular Techniques for Word Embeddings

- **Word2Vec**:
  - Predicts the context of a word given the target word or vice versa.
  - Based on the idea that words in similar contexts have similar meanings.
  
### Dimensionality of Word Embeddings

- Embedding dimensions can range from 1 to thousands:
  - Higher dimensionality captures nuanced relationships but reduces computational efficiency.
- Visualization often limits dimensionality to two or three dimensions for interpretability.

### Pretrained vs. Custom Embeddings in LLMs

- Pretrained models like Word2Vec generate embeddings for machine learning models.
- LLMs create their own embeddings, optimized during training for specific tasks and data.
- LLM embeddings are often high-dimensional and contextualized.
- GPT-2 embedding size: 768 dimensions for smaller models.
- GPT-3 embedding size: 12,288 dimensions for the largest model.

### Preparing Embeddings for LLMs

- Steps include:
  1. Splitting text into words.
  2. Converting words into tokens.
  3. Turning tokens into embedding vectors.

## 2.2 Tokenizing Text

### Overview

Tokenizing text is a critical preprocessing step for creating embeddings for large language models (LLMs). This involves splitting input text into tokens, which may be individual words or special characters, such as punctuation.

- **Input Text**: A public domain short story by Edith Wharton, *The Verdict*, was used for demonstration.
- **Text Source**: Available at [Wikisource](https://en.wikisource.org/wiki/The_Verdict).
- **Sample Size Considerations**: For educational purposes, a single text sample suffices, but real-world LLM training often involves millions of documents.

### Tokenization with Python

Using Python’s `re` module, we can tokenize text as follows:
1. **Initial Tokenization**: Split text into words and punctuation using `re.split`.
2. **Improvement**: Adjust the regular expression to handle a wider variety of special characters, including commas, periods, question marks, and double dashes.
3. **Whitespace Handling**: Whitespace can be preserved or removed based on application needs. For simplicity, whitespace is removed in this example.

- Using the enhanced regular expression, tokens like:['Hello', ',', 'world', '.', 'This', ',', 'is', 'a', 'test', '.'] are successfully extracted from sample text.


## 2.3 Converting Tokens into Token IDs

### Vocabulary Creation
To map tokens to token IDs:
1. **Build Vocabulary**: Extract unique tokens and assign each a unique integer.
2. **Vocabulary Size**: For the short story, the vocabulary contains 1,159 unique tokens.

### Mapping Process
- Tokens are converted to IDs using a mapping dictionary (`vocab`).
- An inverse dictionary enables conversion of token IDs back to tokens.

### Implementation
The process was encapsulated in a `SimpleTokenizerV1` class with:
- **`encode` Method**: Converts text into token IDs.
- **`decode` Method**: Converts token IDs back into text.

### Limitations
The tokenizer raises an error for unknown tokens that are not part of the training vocabulary. For example:
KeyError: 'Hello'

## 2.4 Adding Special Context Tokens

### Enhancements
To handle unknown words and contextualize independent text sources:
1. **New Tokens**:
   - `<|unk|>`: Represents unknown words.
   - `<|endoftext|>`: Marks boundaries between unrelated text sources.
2. **Vocabulary Update**: Incorporates the new tokens, increasing the vocabulary size to 1,161.

- **Unknown Words**: Avoids errors by mapping unknown tokens to `<|unk|>`.
- **Contextual Understanding**: Enhances training by marking boundaries in concatenated text sources.
- The `SimpleTokenizerV2` class extends the previous tokenizer with Incorporates `<|unk|>` and `<|endoftext|>` into the encoding and decoding process.

## Notes: Byte Pair Encoding (BPE) and Data Sampling

### 2.5 Byte Pair Encoding (BPE)
- **Introduction to BPE**:
  - A tokenization technique used to handle unknown words by splitting them into smaller, manageable subword units or individual characters.
  - Frequently used in large language models like GPT-2, GPT-3, and ChatGPT.

- **Benefits of BPE**:
  - Reduces the size of the vocabulary by breaking down rare or unknown words into subwords.
  - Allows the model to generalize better, handling previously unseen words through known subword components.

- **BPE Process**:
  1. Start with a vocabulary of individual characters.
  2. Iteratively merge the most frequent pair of characters or subword units into a new subword.
  3. Continue merging until a predefined vocabulary size is reached or no more merges are possible.

## 2.6, 2.7 Sections are covered in the code file.

## 2.8 Encoding Word Positions in Large Language Models (LLMs)
- **Challenge of Position Awareness**:
  - Token embeddings are generated based on token IDs, but these embeddings are position-agnostic, meaning the same token ID always corresponds to the same vector, regardless of its position in a sequence.
  - The self-attention mechanism in LLMs also lacks an inherent understanding of the order or position of tokens.

- **Need for Positional Information**:
  - To provide position-awareness, two types of positional embeddings are used:
    1. **Absolute Positional Embeddings**: 
       - Each token in the sequence is associated with a unique positional embedding.
       - These embeddings are added to the token embeddings to convey the token's exact position within the sequence.
    2. **Relative Positional Embeddings**:
       - Focuses on the relative distance between tokens, rather than their exact positions.
       - This allows the model to generalize better to sequences of varying lengths.

  - OpenAI's GPT models use **absolute positional embeddings** that are optimized during training.
  - In contrast to predefined positional encodings in the original Transformer model, GPT models optimize these during the training process.

- **Practical Example with Embedding Layer**:
  - A token embedding layer is initialized with a vocabulary size and output dimension. For example:
    - `output_dim = 256` (embedding size)
    - `vocab_size = 50257` (vocabulary size)
  - A data loader samples data and each token is embedded into a 256-dimensional vector.
  
  - **Embedding Output**:
    - The token IDs are converted into a tensor with dimensions (batch_size, sequence_length, embedding_dim), for example, an 8x4x256 tensor.

- **Creating Positional Embeddings**:
  - A separate embedding layer for positional information is created:
    - `context_length = max_length` (maximum sequence length).
  - This produces a tensor of positional embeddings, which are then added to the token embeddings.
  
- **Combining Token and Positional Embeddings**:
  - The positional embeddings are added to the token embeddings, resulting in an input embedding tensor that contains both the token information and its positional information.
  - The final shape of the combined embeddings tensor is also (batch_size, sequence_length, embedding_dim).
