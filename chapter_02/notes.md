# Working with Text Data

## Understanding Word Embeddings
- Why word embeddings?
  - Words are categorical data, and it isn't compatible with the mathematical operations used in deep learning.
  - Therefore, we need a way to represent words as continuous-valued vectors.
- Embedding : Converting data into a continuous-valued vector.
  - Primary objective is to map discrete data to continuous space.
  - Audio, image, and video all need embedding to be processed by deep learning models.
  - Each modality has its own embedding techniques and models.
  - Even for text data, we have different embedding techniques for words, sentences, and documents.
- Steps involved :
  - Splitting text into words.
  - Converting words into tokens.
  - Turning tokens into embedding vectors.
- The model we use for embedding can be learned as a part of pre-training, and their dimension is a design choice.
  - Larger embeddings can encode more contextual information (important given the huge and diverse dataset).
  - Smaller embedding are more computer efficient.
  - Choice of embedding dimension is a trade-off between performance and efficiency.
- Pretrained models like Word2Vec generate embeddings for machine learning models.
- LLMs create their own embeddings, optimized during training for specific tasks and data.
  - LLM embeddings are often high-dimensional and contextualized.
  - GPT-2 embedding size: 768 dimensions for smaller models.
  - GPT-3 embedding size: 12,288 dimensions for the largest model.

## Tokenizing Text
- Tokens : An instance of a sequence of characters in some particular document that are grouped together as a useful semantic unit for processing.
    - Tokenization : Process of splitting text into tokens, and assigning numeric identifier (token IDs) to them.
    - Token IDs are shared by all the instances of the tokens.
    - Embedding model maps the token IDs to continuous-valued vectors.
- Food for thoughts : 
  - Should we make all text lower case ? Capitalization helps LLMs to :
    - Distinguish between proper nouns and common nouns.
    - Understand sentence structure.
    - Learn to generate text with proper capitalization.
  - Should we strip white-spaces or have it as a separate token ? 
    - Keeping whitespaces can be useful if we train models that are sensitive to the exact structure of the text.
    - Example : Python code sensitive to indentation and spacing.

## Converting Tokens into Token IDs
- Vocabulary : A set of unique tokens in the dataset (is often sorted alphabetically).
- To create a vocabulary, we extract unique tokens from the text data.
- Next step after creating tokens for text data is to convert them (tokens) into token IDs.
  - Token IDs are unique numeric identifiers for each token
  - To do this step we need to develop a vocabulary first.
  - Process of converting tokens into token IDs is called `numericalization`.
  - Process of converting token IDs back to tokens is called `denumericalization`.
  - Hashmaps are used to map tokens to token IDs and vice versa.
- If the vocabulary doesn't contain a token, and we try to convert it into token ID, we get a `KeyError`.
- Handling punctuations, whitespaces, special characters and unknown words in the vocabulary is important.

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

## Data Sampling with a Sliding Window
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

## Creating Token Embeddings
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

## Encoding Word Positions
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
