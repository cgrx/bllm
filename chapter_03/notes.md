# Coding Attention Mechanisms

## Overview
- The overall goal is to understand how attention mechanisms work. We will do it by coding simplified versions.
  - `simplified self-attention mechanism` to understand how attention mechanisms work.
  - `self-attention mechanism` where the weights are learned during training.
  - `causal attention` where data snooping is prevented by depending only on the past and current tokens.
  - `multi-head attention` where representations from multiple subspaces are attended to at the same time.
- Understanding attention mechanism is important to code GPT model in Chapter 4.

##  The problem with modeling long sequences
- RNNs have an encoder-decoder architecture.
- Encoder : Maps input sequence to a fixed length vector.
  - Using recurrence relations, the encoder maps the input sequence to a fixed length vector.
  - The fixed length vector contains the contextual information of the entire input sequence.
- Decoder : Maps the fixed length vector to the output sequence.
  - The decoder does not have access to the input sequence / intermediate states of the encoder.
  - The initial hidden state of the decoder is the fixed length vector from the encoder.
  - The decoder updates its hidden state and generates the output sequence one token at a time.
- When we are doing sequence-to-sequence tasks for long texts, we don't have direct access to previous tokens.
- Reliance on a single fixed length vector to encode entire contextual information causes informational bottleneck.
- Additionally, encoding using recurrence relations is computationally expensive as it can't be parallelized.
- As a result, models like RNNs don't work well for long sequences.

## Capturing data dependencies with attention mechanisms
- Due to limitations of RNNs, discussed above, it doesn't work well for long sequences.
  - To address this, Bahdanau attention mechanism (BAM) was proposed.
  - BAM allow the decoder to selectively focus on different parts of the input sequence at each decoding step.
- Transformer architecture was proposed to address both the limitations of RNNs.
  - The self-attention mechanism was inspired by Bahdanau attention mechanism.
  - The self-attention mechanism allows the model to capture dependencies between different parts of the input sequence.
  - How ? It allows each position in the input sequence to interact and weigh the relevancy of all other positions.
  - The transformer architecture is parallelizable and can be trained faster than RNNs.

## Attending to different parts of the input with self-attention
- Self-attention : Mechanism captures dependencies between different positions within an input sequence.
- The dependencies are captured by computing attention weights for a given position against all other positions.
- The weights in the attention mechanism are learned during training.
- Why the word `self` ?
  - `Self` in self-attention refers to the fact that the attention mechanism is applied to the input sequence itself.
  - This is in contrast this with Bahdanau attention mechanism, where weights capture dependencies between input and output sequences.
- Simplified self-attention mechanism :
  - Let $X \in \mathbb{R}^{T \times D}$ where $X_{i*}$ is the token embedding for position $i$.
  - Let $\omega := XX^{T} \in \mathbb{R}^{T \times T}$ be the attention scores matrix.
    - $\omega_{ij}$ is the dot product of the token embeddings for positions $i$ and $j$.
    - The dot product captures the similarity between the token embeddings.
  - Let $A := \text{softmax}(\omega) \in \mathbb{R}^{T \times T}$ be the attention matrix.
    - The softmax function normalizes the attention scores to sum to 1.0 for each row of $\omega$.
    - $A_{i*}$ represents the relative importance of each token to the $i^{th}$ token. x.
  - $Z := AX \in \mathbb{R}^{T \times D}$ where $Z_{i*}$ is the context vector for the $i^{th}$ position.

## Implementing self-attention with trainable weights
- Let $X \in \mathbb{R}^{T \times D}$ where $X_{i*}$ is the token embedding for position $i$.
- Instead of using $X$ directly, we will project $X$ into query, key, and value matrices.
  - Query ($q_i$) : Current token that model is focusing on.
  - Key ($k_j$) : Token that model is comparing the current token to.
  - Value ($v_j$) : The value of the token that model is comparing the current token to.
  - First the model understand the relative importance of each token to the current token.
  - Based on the importance, the model computes the context vector for the current token using the value vectors.
- Let $W_q, W_k, W_v \in \mathbb{R}^{D \times D}$ be trainable weight matrices.
  - $Q = XW_q \in \mathbb{R}^{T \times D}$ is the query matrix, where $Q_{i*}$ is the query vector for the $i^{th}$ token.
  - $K = XW_k \in \mathbb{R}^{T \times D}$ is the key matrix, where $K_{i*}$ is the key vector for the $i^{th}$ token.
  - $V = XW_v \in \mathbb{R}^{T \times D}$ is the value matrix, where $V_{i*}$ is the value vector for the $i^{th}$ token.
- Let $\omega := QK^{T} \in \mathbb{R}^{T \times T}$ be the attention scores matrix.
  - $\omega_{ij}$ is the dot product of the query vector for position $i$ and the key vector for position $j$.
- Let $A := \text{softmax}(\omega / \sqrt{D}) \in \mathbb{R}^{T \times T}$ be the attention matrix. (based on the comments @ [LINK](https://ai.stackexchange.com/a/42197))
  - Why $\sqrt{D}$ ?
    - Let's assume that $Q$ and $K$ are independent with zero mean and unit variance.
    - $\omega_{ij} := Q_{i} \cdot K_{j} = \sum_{d=1}^{D} Q_{id} \cdot K_{jd}$.
    - $var(\omega_{ij}) = \sum_{d=1}^{D} var(Q_{id} \cdot K_{jd}) = \sum_{d=1}^{D} var(Q_{id}) \cdot var(K_{jd}) = D.$
    - Standard deviation $\sigma(\omega_{ij}) = \sqrt{D}$.
  - Why scale by $\sqrt{D}$ ?
    - $var(\omega_{ij}) = D$ implies, the dot products grows larger in magnitude as $D$ increases.
    - As the dot product increases, softmax behaves like a step function.
      - One element of the attention score is going to be greater than the others.
      - The relative difference gets amplified by the softmax function.
      - This implies, softmax is going to assign a probability closer to 1.0 for one element and closer to 0.0 for the others.
      - That is leads to step function like behavior. 
    - As the softmax saturates, and behaves like a step function, the gradients become very small.
    - Small gradients make it difficult to train the model.
    - To counteract this, we scale the dot products by its standard deviation (i.e.) $\sqrt{D}$. 
  - The softmax function normalizes the attention scores to sum to 1.0 for each row of $\omega / \sqrt{D}$.
  - $A_{i*}$ represents the relative importance of each token to the $i^{th}$ token.
  - Note : 
    - We will be using layer normalization to ensure that the query and key vectors have zero mean.
    - We will be using Xavier initialization to ensure that the query and key vectors have unit variance.
- $Z := AV \in \mathbb{R}^{T \times D}$ where $Z_{i*}$ is the context vector for the $i^{th}$ position.

## Hiding future words with causal attention
- Causal attention : Mechanism that prevents data snooping by depending only on the past and current positions.
- Causal attention is used for tasks where the model should not have access to future tokens.
- Naive way of implementing causal attention :
  - Let $A \in \mathbb{R}^{T \times T}$ be the attention matrix.
  - Let $M := \text{Tril}(1; 0) \in \mathbb{R}^{T \times T}$ be a lower triangular matrix with ones on and below the diagonal.
  - The masked attention matrix (un-normalized) is $\tilde{A} := AM$.
  - We need to re-normalize the rows of $\tilde{A}$ to sum to 1.0 (i.e.) $\bar{A} := \text{softmax}(\tilde{A})$.
  - $Z := \bar{A}V \in \mathbb{R}^{T \times D}$ where $Z_{i*}$ is the context vector for the $i^{th}$ position.
- Efficient way of implementing causal attention :
  - Insight : $\lim_{x \to \infty} e^{-x} = 0$.
  - Let $M := L + U \in \mathbb{R}^{T \times T}$.
    - $L$ is the lower triangular matrix with ones on and below the diagonal.
    - $U$ is the upper triangular matrix with $-\inf$ in elements above the diagonal, and zeros elsewhere.
    - $L := \text{Tril}(1;0)$ and $U := \text{Triu}(-\infty; 1)$.
  - The masked attention score is $\tilde{\omega} := \omega M$.
    - We are selecting only the values for the past and current positions.
    - The future positions are masked with $-\infty$. In softmax, this will be 0.
  - $A := \text{softmax}(\tilde{\omega} / \sqrt{D}) \in \mathbb{R}^{T \times T}$ is the masked attention matrix.
  - $Z := AV \in \mathbb{R}^{T \times D}$ where $Z_{i*}$ is the context vector for the $i^{th}$ position.
- Applying dropout
  - Dropout is applied to prevent reliance on only few specific hidden units over others.
  - This is applied at training time only.
  - Two places where dropout can be applied : Attention weights and context vectors.
  - The dropout factor $p \in [0, 1]$. 
    - Corresponding fraction of elements is dropped from the matrix by setting them to zero.
    - To compensate for the dropped elements, the remaining elements are multiplied by $1 / (1 - p)$.

## Extending single-head attention to multi-head attention
- A self-attention module is called an `attention head`.
- Multi-head attention : 
  - Multiple self-attention heads are used to capture dependencies from different subspaces.
  - Each head learn different representations of the input sequence, independently, in parallel.
- The outputs of the heads are concatenated to get the final output.
  - Let $H$ be the number of attention heads. 
  - Let $Z_i \in \mathbb{R}^{T \times D}$ be context matrix for the $i^{th}$ head.
  - $Z := \text{concat}(Z_1, Z_2, \cdots, Z_H) \in \mathbb{R}^{T \times HD}$.
- Efficient way of implementing multi-head attention :
  - Let $X \in \mathbb{R}^{T \times D}$ where $X_{i*}$ is the token embedding for position $i$.
  - Let $H$ be the number of attention heads.
  - Let $W_Q, W_K, W_V \in \mathbb{R}^{D \times HD}$ be weight matrices for the query, key, and value projections.
  - Let $P \in \mathbb{R}^{HD \times HD}$ be a matrix for output projection.
  - We need transformation functions to split and transpose the input to the desired shape.
    - $S : \mathbb{R}^{T \times HD} \to \mathbb{R}^{T \times H \times D}$ be a function that splits the input into $H$ heads.
    - $\bar{S} : \mathbb{R}^{T \times H \times D} \to \mathbb{R}^{T \times HD}$ be a function that reverse $S$. 
    - $T_{12} : \mathbb{R}^{T \times H \times D} \to \mathbb{R}^{H \times T \times D}$ be a transpose along the first and second dimensions.
    - $T_{23} : \mathbb{R}^{H \times T \times D} \to \mathbb{R}^{H \times D \times T}$ be a transpose along the second and third dimensions.
  - The key, value, and query matrices are computed as follows :
    - $K := S(XW_K)^{T_{12}} \in \mathbb{R}^{H \times T \times D}$
    - $V := S(XW_V)^{T_{12}} \in \mathbb{R}^{H \times T \times D}$
    - $Q := S(XW_Q)^{T_{12}} \in \mathbb{R}^{H \times T \times D}$
  - Attention score is $\omega = QK^{T_{23}} \in \mathbb{R}^{H \times T \times T}$.
  - Let $M := L + U \in \mathbb{R}^{H \times T \times T}$ where 
    - $L \in \mathbb{R}^{H \times T \times T} : L_h = \text{Tril}(1;0) \in \mathbb{R}^{T \times T}, \forall h \in \{1, \cdots, H\}$ 
    - $H \in \mathbb{R}^{H \times T \times T} : U_h = \text{Triu}(-\infty; 1) \in \mathbb{R}^{T \times T}, \forall h \in \{1, \cdots, H\}$.
  - Masked attention score is $\tilde{\omega} := \omega M \in \mathbb{R}^{H \times T \times T}$.
  - Attention matrix is $A := \text{softmax}(\tilde{\omega} / \sqrt{D}) \in \mathbb{R}^{H \times T \times T}$ where we normalize along the last dimension.
  - Let $\tilde{A} := \text{Dropout}(A; p) \in \mathbb{R}^{H \times T \times T}$ where $p \in [0,1]$, be the attention matrix with dropout applied.
  - The context matrix is $Z := \tilde{A}V \in \mathbb{R}^{H \times T \times D}$.
  - The output is $Y := \bar{S}(Z^{T_{12}})P^{T} \in \mathbb{R}^{T \times HD}$
