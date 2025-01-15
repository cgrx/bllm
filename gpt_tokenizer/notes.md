# BPE Tokenization - Andrej Karpathy

- What is tokenization
    
    Convert raw string to a sequence of integers or codes. Individual characters to integers. Being able to Encode and decode (refer to basics in Intro to LLM page)
    
- what are the tokenization used by orgs
    - Tiktoken - BPE tokenization strategy.

The primary idea for this notebook and code along with it is to build a tokenizer from scratch.

## Introduction

- What are the complexities of tokenization that it creates a lot of probs?
    - Reasons like LLMs cant spell words
    - when llms cacnt do super simple string processing tasks like reversing a string
    - when they are worse at non enlgish languages
    - when it is bad at simle arithmetic
    - when they have more necessary trouble coding in python.
    - 
    
    ![image.png](image.png)
    

https://www.google.com/url?q=https%3A%2F%2Ftiktokenizer.vercel.app - Good tokenizer example

- How does LLMs take in tokens
    
    Sometimes numbers break up in weird different ways like 127can be taken as a single token but 677 might be two tokens adn 127 + 677 can result in weird ansewrs. Egg by itself can be two tokens but just ‘ Egg’ can be a different token. EGG can also be different tokens. ffor the same concept, we can have very different tokens and Ids. The LLm has to learn that these are all the same concept and have to group them together. 
    
- WHy LLms perform worse in non english languages?
    - Lack of data. Tokenization also performs poorly because no. of tokens used for other languages are much larger. Tokens for teh same translation are broken down into smaller components so its loaded up with lot of tokens, LLms is stretched out with other languages, and have a lot of other boundaries.
- How does it react to code?
    - each white space is considered as a single token. but gradually they have ensured that they combined white spaces together. to have a less no. of tokens. Lot of improvement for LLms comes from designing how to improve tokenizers.

RQ: Is there a way to completely delete the need for tokenization?

## Strings and Unicodes

- Strings are by definition  - series of unicode code points. I is a way to define types of characters.
- we can access unicode points by using ord() function in python. chr() is opposite of ord().
- A code point is only for a character which gives you an integer.
- To look for code points for a string

```python
[ord(x) for x  in "sample sentence"]
```

- Why cant we use these integers and not have any tokenization at all?
    - Vocabulary would be quite long - unicode vocab - 150,000 codepoints +
    - Code points is very much alive and keep changing.
- Hence we turn towards encoding. UTF-8, UTF-16,UTF-32. What they do is take a unicode integer and translate it to binary encodings (byte strings). UTF-8 translates unicodes to 1-4 bytes.

[Read on Unicodes -](https://www.reedbeta.com/blog/programmers-intro-to-unicode/) Programmer’s introduction to Unicodes.

- String presenting at utf-8 encoding

```python
list("안녕하세요 👋 (hello in Korean!)".encode("utf-8"))
```

HOwever even if we use UTF-8 naively, all of our text would be stretched out with very long sequence of bytes. Our sequences are very long.  We only have as much context length but we hvae large sequences so it wont help our purpose. So what we do? we turn to Byte Pair encoding algorithm. 

There is a possibility to directly feed in bytes to transformer architecture, that can just feed in raw bytes. There has been a paper () meaning tokenization free feeding into transfomer models.

## Byte Pair Encoding

Basic Idea: finding the pair of characters that occurs more frequently, then we just pair them up and create a new token. This way we can reduce the no. of tokens. 

Steps for BPE :

- Take Raw text and convert to UTF-8
- Find the most occuring pairs. ( hint: one way is to iterate over pairs and make a dictionary value key list. using zip function)
- Iterate over the entire list again, and everytime the most occuring pair occurs, we replace the unicode pair and update it with 256 ( since UTF-8 goes from 0-255).
- We can iterate this process multiple times. We can decide an hyperparameter to decide on amount of times we can repeat such that we can increase the vocabulary but decrease the sequence length.

### Tokenization is a complete different stage.

Tokenizer is a complete seperate object from the Large Language Model. We use the BPE algorithm to train the Tokenizer, it has a seperate training set etc. Once the Tokenization is done, we can do encoding and decoding. It can take raw text and translate it back to token sequence and vice versa.

![image.png](image%201.png)

## Encoding and Decoding

Given the sense of integeers from [0,vocab_size], what is the text?

- We can use UTF-8 Decode from its library. Standard practice to ensure that it follows UTF-8 format is use errors = “replace” parameter.

```python
text = token.decode("utf-8", errors = "replace")
```

- For Encoding , first you can directly encode the text to raw bytes and convert it into list of integers. And Now ofc, we can use the merges dictionary (where merges were built using the maximum occuring pairs), and its also built from top to bottom( order is important and is preserved). we can reuse the keys from the dictionary, ( we have to take the keys that have the lowest index) . He uses min function here

```python
pair = min(stats, key= lambda p: merges.get(p, float("inf")))
```

- The reason for putting a float inf, here if we come across a pair that doesnt exist, then we just mention float inf so that it wont participate while using min operation. This function will fail in way such that if there is nothing to merge and everythign just returns float inf, and the pair will become the first element of the stats. This is just trying out the fancy way in python.

## GPT-2 Implementation of BPE - Force Splits across categories.

- The GPT-2 Paper ensures that phrases like “dog.” or “dog?”  using a complex regex pattern

```python
gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

print(re.findall(gpt2pat, "Hello've world123 how's are you!!!?"))
```

- The tokenizer first splits your text into list of texts.
- This regex pattern is hardcoded - they dont cater towards unicode ‘ as well as capitalized phrases.
- cl100k_base is the tokenizer base used for gpt4.
- Gpt 2 doesnt merge spaces, while gpt4 merges spaces.
- https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py - the tiktoken base for all kinds of tokenizers by open ai. It has all the regex patterns that split the text before encoding.
- The major changes in gpt4 is that is caseinsensitive, different handling of white space, they chooose to match 1-3 numbers, they will never merge more than that ( we dont really know why they do this stuff).

### GPT-2 Encoder.py

[https://github.com/openai/gpt-2/blob/master/src/encoder.py](https://github.com/openai/gpt-2/blob/master/src/encoder.py)

- The two key components in a tokenizer is merges and vocab. Open Ai also has a byte enocder and byte decoder. What they do is byte encode and tehn encode, and byte decode and then decode.
- Algorithmicially its  identifical to what we built so far.

### Special Tokens

- The length of encoder of gp2 - is 50257. how?
- 256 raw byte tokens. 50,0000 merges. +1 special token . “endoftext”

```python
encoder['<|endoftext|>'] # the only special token in use for the GPT-2 base model
```

- This handles special case instructions to particularly encode special tokens.
- There are finetuned and base tokenizers. similar to how pretrained and finetned models are. You can extend tiktoken, meaning you can take the base tokenizer and you can add special tokens.
- In Gpt 4 we have 4 more additiona tokens FIN_prefix, FIN_middle-Fin_suffic, endofprompt. The idea is in this paper https://arxiv.org/pdf/2207.14255
- The common practice - train  a language model, We can add special tokens later. It is like a surgery that we are adding an additional final vector and integers and wee have to modify the embedding dimensions or share it in the last layer.

### MinBPE exercise

- Building your own GPT-4 Tokenizer that you can match with the tiktoken GPT4 Code base. Ensure the vocabulary of boith is the same, in a way, using a dataset like wikipedia page on your own.

### SentencePiece Tokenization

- used by both Llama and Mistral series.
- Biggest difference between sentencepiece and tiktoken - it runs PBE on the unicdoe points directly and then if it has an option of character coverage for what to do with very rare codepoints that appear very rare few times, it either mpas tem onto an UNK token, or if byte_fallback is turned on, it encodes them with utf-8 and then encodes the raw bytes instead.
- https://github.com/google/sentencepiece
- There are tons of options and configurations, a bit more complicated than tiktoken, also because it is existing for a long time.
- Sentencepiece has the concept of sentences. It has the no. of sentences, max sentence length etc. For it sentences are like individual training example. But in context of LLms are irrelvavnt cuz its difficult to define what an actual sentence is. There are different concepts for it in differnet languages.

```python
print([sp.id_to_piece(idx) for idx in ids])

['▁', 'h', 'e', 'l', 'lo', '▁', '<0xEC>', '<0x95>', '<0x88>', '<0xEB>', '<0x85>', '<0x95>', '<0xED>', '<0x95>', '<0x98>', '<0xEC>', '<0x84>', '<0xB8>', '<0xEC>', '<0x9A>', '<0x94>']
```

```markdown
This is why we have bytefallback, if there is some unrecognized character, it encodes to check with utf-8 and directly uses that to encode that character information

- Why do we have an extra space in front. Its coming from addw_prefix = True.'world' in itself has a different id from '_world'. Words in the beginning of sentences and words in the middle of sentences can be completely diferent. So somemtimes its better to add a space before the first word, so that we have both worlds as the same.
```

### Vocab - size

- What should be the vocab size?
    - vocab size is used basically important in the first layer which is the vector representation for each token which is modified using backpropagation. If we have more and more tokens we need to have more and more probabilities. Every single token introduces additional dot products in the linear layer a well.
    - If we hvae very large vocabulary size, every one of the tokens are going to come more and orme rarely, we will see fewer and fewer examples for each indvidiual tokens and we might assume it will be undertrained. On the other hand, if we hvae really small no. of vocab, we are squising too large of a text into small tokens.  it is an imperical hyperparameter. Currently its 10k-100k today.
- You can totally add a token if possible. We have to resize and calculate probabilties for new tokens. You can freeze the basemode, introduce new parameters and only train that .
- There is entire fieldof design space. https://arxiv.org/pdf/2304.08467

```markdown
- Why can't LLM spell words? **Tokenization**. 
Because how it splits and defines tokens. It finds it quite difficult to do character level tasks.

- Why can't LLM do super simple string processing tasks like reversing a string? **Tokenization**.
- Why is LLM worse at non-English languages (e.g. Japanese)? **Tokenization**.
- Why is LLM bad at simple arithmetic? **Tokenization**.
- Why did GPT-2 have more than necessary trouble coding in Python? **Tokenization**.
- Why did my LLM abruptly halt when it sees the string "<|endoftext|>"? **Tokenization**.
- What is this weird warning I get about a "trailing whitespace"? **Tokenization**.
- Why the LLM break if I ask it about "SolidGoldMagikarp"? **Tokenization**.
- Why should I prefer to use YAML over JSON with LLMs? **Tokenization**.
- Why is LLM not actually end-to-end language modeling? **Tokenization**.
- What is the real root of suffering? **Tokenization**.
```