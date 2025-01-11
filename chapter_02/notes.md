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

## Adding Special Context Tokens
- We can add an `<|unk|>` token to deal with Out-Of-Vocabulary (OOV) words.
- The text documents are concatenated for training. Therefore, we can use `<|endoftext|>` token to separate them.
- Traditionally, we use other special tokens as well :
  - `<|startoftext|>` token to indicate the beginning of a text document.
  - `<|pad|>` token to pad the text to a fixed length.
- In modern LLMs, we use `<|endoftext|>` token for padding as well.
  - We use masking to ignore the padding tokens during training.
  - Therefore, specific token used for padding doesn't matter much.
- Additionally, modern LLMs don't use `<|unk|>` token.
  - They use sub-word tokenization techniques.
  - Sub-word tokenization techniques can handle OOV words better than the `<|unk|>` token.
  - Example : Byte Pair Encoding (BPE).

## Byte Pair Encoding
- Byte Pair Encoding (BPE) is a tokenization technique that :
  - Handles unknown words by breaking them down into subword units or individual characters.
  - It was used in the training of GPT-2, GPT-3, and initial version of ChatGPT.
- BPE implementation via `tiktoken` library :
  - The vocabulary size (gpt-2) is 50,257.
  - `<|endoftext|>` token has the token ID of 50256 (last token ID).
  - It encodes and decodes unknown words (for example : `someunknownPalace`).
- How does BPE handle unknown words?
  - It breaks down the unknown word into sub-words (might even be individual characters) that are in the vocabulary.
  - In decoding, it combines the sub-words to form the unknown word.
- How does BPE build the vocabulary ?
  - Detailed discussion is a bit out-of-scope.
  - Vocabulary is build by iteratively merging frequent characters into sub-words and frequent sub-words into words.
    - BPE starts with adding all individual single characters to its vocabulary ("a", "b", ...). 
    - In the next stage, it merges character combinations that frequently occur together into sub-words.  
    - The merges are determined by a frequency cutoff.
  - For example, "d" and "e" may be merged into the sub-word "de".

## Data sampling with a sliding window
- Given a large text dataset, we can now tokenize it and convert the tokens into token IDs.
- The next step is to generate an input-target pair for training.
- In our case, we do it in self-supervised manner.
  - Let the length of the input sequence be `N` (number of tokens).
  - Let the context window for the model be `W` (max number of tokens in the input per sequence).
  - Let the stride be `T` (can be $>=1$).
  - First sequence :
    - Input : Tokens $\{t_1, ..., t_W\}$
    - Target : Tokens $\{t_2, ..., t_{W+1}\}$
  - Second sequence :
    - Input : Tokens $\{t_{T+1}, ..., t_{T+W+1}\}$
    - Target : Tokens $\{t_{T+2}, ..., t_{T+W+2}\}$
  - And so on. Basically, we slide the window by `T` token at a time.
  - The context window size `W` is the sliding window size.
- Often the `T` (stride length) is set to `W` (context window size) for the following reasons :
  - Doesn't skip any part of the dataset.
  - Minimizes overlap as it contributes to overfitting.

## Creating token embeddings
- Next step is to convert each batch of sequences into token embeddings.
- Token embeddings are continuous-valued vectors that represent the tokens.
  - The embedding matrix should be initialized with random weight at the start of training.
  - The embedding dimension `D` is a hyperparameter.
  - If the vocabulary size is `V`, then the embedding matrix is of size `V x D`.
- For every token `t` in the sequence :
  - We look up the token ID in the embedding matrix to get the token embedding.
  - The token ID corresponds to the row index in the embedding matrix.
  - We can use matrix multiplication to get the token embeddings efficiently for a batch of sequences.
- If we have a batch of sequences :
  - Let `W` be context window size, `B` be batch size, `D` be embedding dimension.
  - The input tensor is of size `B x W` and the output tensor is of size `B x W x D`.
- The embedding matrix is learned (due to flow of gradients during back-propagation).

## Encoding word positions
- Challenge of position awareness:
  - Token embeddings are generated based on token IDS.
    - Two tokens with the same ID will have the same embedding, regardless of their position in the sequence.
    - This implies that these embeddings are position-agnostic.
  - The self-attention mechanism in LLMs also lacks an inherent understanding of the order or position of tokens.
- Need for positional information:
  - Model needs to understand the order and relationship between tokens to generate coherent and context-aware text.
  - To make the model position-aware, we can add position embeddings to the token embeddings.
- Types of position embeddings:
  - Absolute Position Embedding: 
    - Each position has a unique embedding and is added to the token embedding.
    - This favours the model to learn "at which exact position" the token is in the sequence.
  - Relative Position Embedding:
    - The relative distance between tokens is encoded in the position embedding.
    - This favours the model to learn "how far" the tokens are from each other.
  - The choice of position embedding is application dependent. 
  - However, relative position embedding is favoured as it can generalize better to unseen sequence lengths.
- Examples:
  - The original Transformer model uses predefined positional encodings.
  - OpenAI's GPT models use Absolute Positional Embeddings that are optimized during training.
- Mechanics of encoding positional information:
  - Let context window size be `W`, and embedding dimension be `D`.
  - Position embedding matrix is of size `W x D`.
  - To each sequence in a batch of batch size `B`, we add the position embedding.
  - The output tensor is of size `B x W x D`.
