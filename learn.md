# TinyChatGPT Learning Guide

**Purpose**: A living glossary + mental model you'll keep updating as the project grows.
If you truly understand this file, transformers stop feeling like magic.

---

## 1. What is a Model?

A model is a function with parameters (numbers) that maps input → output.

**For a chatbot**:  
`(text so far) → (probabilities for next token)`

**Key Insight**:  
The model does not "know answers".  
It only learns patterns of continuation.

---

## 2. Token

A token is the smallest unit the model sees.

**Examples**:
- Character-level: "a", "b", "?"
- Token-level: "hello", "world", "ing"

**Important**:  
Models don't see words. They see numbers representing tokens.

---

## 3. Vocabulary (vocab)

The vocab is the set of all possible tokens.

**Sizes**:
- char-level vocab ≈ 50–100
- BPE vocab ≈ 5k–50k+

**Tradeoff**:
- Small vocab → longer sequences
- Large vocab → more parameters

---

## 4. Embedding

An embedding converts a token ID into a vector.

**Example**:  
`token_id (7) → [0.21, -0.9, 1.2, ...]`

**Why?**:  
Because neural networks work with numbers, not symbols.

---

## 5. Sequence Length (Context Length)

The number of tokens the model can "see" at once.

**Example**:  
Context length = 64 → model only remembers the last 64 tokens

**Warning**:  
Attention cost grows as O(T²). This is why we start small.

---

## 6. Autoregressive Modeling (VERY IMPORTANT)

GPT-style models predict one token at a time.

**Formula**:  
`P(tₙ | t₁, t₂, ..., tₙ₋₁)`

**During chat**:
1. User types prompt
2. Model predicts next token
3. That token is appended
4. Repeat

This is why it feels like "thinking".

---

## 7. Decoder-only Transformer

This is the exact family ChatGPT belongs to.

**Characteristics**:
- No encoder
- Uses causal self-attention
- Generates tokens left → right

You are building a real architecture, just smaller.

---

## 8. Self-Attention (Intuition)

Self-attention lets each token:

> "Look at other tokens and decide what matters"

**Example**:  
"The animal didn't cross the street because it was tired"  
Attention helps "it" focus on *animal*, not *street*.

---

## 9. Causal Mask

A causal mask prevents cheating.

**Without mask ❌**:  
token 5 can see token 10 (future)

**With mask ✅**:  
token 5 sees only tokens 1–5

**Critical**: If this is wrong → your model is broken.

---

## 10. Transformer Block

One block contains:
1. LayerNorm
2. Self-Attention
3. Residual connection
4. LayerNorm
5. MLP
6. Residual connection

Stack many blocks → intelligence increases.

---

## 11. Residual Connection

Residual = "add input back"

**Formula**:  
`output = F(x) + x`

**Why?**:
- Prevents gradient collapse
- Allows deep networks to train

---

## 12. Loss Function

The loss measures how wrong the model is.

**For GPT**: Cross-Entropy Loss

**Relationship**: Lower loss = better predictions

**Training** = minimize loss.

---

## 13. Backpropagation

Backprop answers:

> "Which parameters caused the error, and how do we fix them?"

**Gradient flow**:  
`loss → logits → layers → embeddings`

---

## 14. Optimizer

The optimizer updates parameters.

**Common choice**: AdamW

**It controls**:
- Learning speed
- Stability
- Convergence

---

## 15. Overfitting

When the model memorizes instead of learning patterns.

**Signs**:
- Training loss ↓
- Validation loss ↑

**Fixes**:
- More data
- Regularization
- Early stopping

---

## 16. Inference vs Training

| **Training**                  | **Inference**                  |
|-------------------------------|--------------------------------|
| Model sees correct answers    | No gradients                   |
| Loss is computed              | Model generates tokens freely  |
| Gradients update weights      |                                |

---

## 17. Checkpoint

A checkpoint saves:
- Model weights
- Optimizer state
- Training step

**This lets you**:
- Resume training
- Deploy models
- Compare versions

---

## 18. Scaling Law (Big-Picture 🔥)

Performance improves with:
- More data
- More parameters
- More compute

But architecture stays the same.

**That's why starting with GPT-style is smart.**

---

## 19. What "Smart" Means Here

Smart ≠ big.

**Smart =**:
- Correct architecture
- Clean data format
- Reproducible training
- Ability to scale later

---

## 🚀 How to Use This Guide

1. **When confused**: Read the relevant section
2. **When stuck**: Ask "which concept am I missing?"
3. **When learning**: Update this file with new insights
4. **When explaining**: Use these analogies

## 📈 Progress Tracking

- [ ] Understand all 19 concepts
- [ ] Build working tokenizer
- [ ] Implement causal attention
- [ ] Train first model
- [ ] Generate coherent text
- [ ] Create working chat interface

---

*Last updated: [Date]*  
*Version: 1.0*