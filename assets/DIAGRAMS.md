# KAMUI — Architecture Diagrams

All diagrams are written in Mermaid. They render in GitHub markdown,
the mkdocs site, and any Mermaid-compatible viewer.

---

## 1. Repository Structure

```mermaid
graph TD
    ROOT[KAMUI/]

    ROOT --> LIB[kamui/]
    ROOT --> CFG[configs/]
    ROOT --> SCR[scripts/]
    ROOT --> NB[notebooks/]
    ROOT --> TST[tests/]
    ROOT --> DOC[docs/]
    ROOT --> RES[research/]
    ROOT --> DATA[data/]
    ROOT --> GH[.github/]

    LIB --> TOK[tokenizer/]
    LIB --> MOD[model/]
    LIB --> TRN[training/]
    LIB --> HKS[hooks/]
    LIB --> MCI[mechinterp/]
    LIB --> EVL[evaluate/]
    LIB --> UTL[utils/]

    MOD --> MC[config.py]
    MOD --> ME[embedding.py]
    MOD --> MA[attention.py]
    MOD --> MF[feedforward.py]
    MOD --> MN[normalization.py]
    MOD --> MB[block.py]
    MOD --> MT[transformer.py]

    MCI --> ML[logit_lens.py]
    MCI --> MAP[activation_patch.py]
    MCI --> MI[induction.py]
    MCI --> MCR[circuits.py]
    MCI --> MAV[attention_viz.py]
    MCI --> MP[probing.py]

    RES --> EXP[experiments/]
    RES --> RPT[reports/]
    RES --> FUT[future/]
    RES --> RL[RESEARCH_LOG.md]
```

---

## 2. Module Dependency Graph

```mermaid
graph LR
    UTL[utils]
    TOK[tokenizer]
    MOD[model]
    TRN[training]
    HKS[hooks]
    MCI[mechinterp]
    EVL[evaluate]

    UTL --> TOK
    UTL --> MOD
    UTL --> TRN
    UTL --> HKS
    UTL --> MCI
    UTL --> EVL

    TOK --> MOD
    TOK --> TRN

    MOD --> TRN
    MOD --> HKS
    MOD --> EVL

    HKS --> MCI

    style UTL fill:#2d2d2d,stroke:#666,color:#fff
    style MCI fill:#1a3a5c,stroke:#4a8cc4,color:#fff
    style HKS fill:#1a3a3a,stroke:#4ac4a8,color:#fff
```

---

## 3. Transformer Architecture

```mermaid
graph TD
    IN["token_ids  (B, S)"]

    IN --> EMB["Embedding
    token + positional
    (B, S, D)"]

    EMB --> RS0["residual stream
    (B, S, D)"]

    RS0 --> BLK["TransformerBlock × n_layers
    ───────────────────────────
    LayerNorm
        ↓
    MultiHeadAttention
        ↓  (+ residual)
    LayerNorm
        ↓
    FeedForward
        ↓  (+ residual)"]

    BLK --> RSN["residual stream
    (B, S, D)"]

    RSN --> FLN["Final LayerNorm"]
    FLN --> UNE["Unembed  D → V"]
    UNE --> OUT["logits  (B, S, V)"]

    style BLK fill:#1a2a4a,stroke:#4a7ac4,color:#fff
    style EMB fill:#2a1a4a,stroke:#7a4ac4,color:#fff
```

---

## 4. Multi-Head Attention

```mermaid
graph TD
    X["x  (B, S, D)"]

    X --> WQ["W_Q  →  Q  (B, H, S, Dh)"]
    X --> WK["W_K  →  K  (B, H, S, Dh)"]
    X --> WV["W_V  →  V  (B, H, S, Dh)"]

    WQ --> SDPA["Scaled Dot-Product Attention
    ─────────────────────────────
    scores = QKᵀ / √Dh   (B,H,S,S)
    masked scores  (causal: -∞ future)
    weights = softmax(masked)
    output = weights × V"]

    WK --> SDPA
    WV --> SDPA

    SDPA --> WTS["weights  (B, H, S, S)  ← returned for interpretability"]
    SDPA --> ATT["attn output  (B, H, S, Dh)"]

    ATT --> CON["Concat heads  →  (B, S, D)"]
    CON --> WO["W_O projection  →  (B, S, D)"]

    style SDPA fill:#1a3a2a,stroke:#4ac47a,color:#fff
    style WTS fill:#3a1a1a,stroke:#c44a4a,color:#ccc
```

---

## 5. Hook System

```mermaid
sequenceDiagram
    participant User
    participant HookManager
    participant Model
    participant ActivationCache

    User->>HookManager: with HookManager(model) as hooks:
    HookManager->>Model: register_forward_hook on blocks.3.attn
    HookManager->>Model: register_forward_hook on embed

    User->>Model: model(token_ids)
    Model->>ActivationCache: cache embed.output
    Model->>ActivationCache: cache blocks.3.attn.weights
    Model->>ActivationCache: cache blocks.3.attn.output
    Model-->>User: logits

    User->>HookManager: hooks.get("blocks.3.attn.output")
    HookManager-->>User: Tensor (B, S, D)

    User->>HookManager: (exit context)
    HookManager->>Model: remove all hooks
    Note over Model: No hooks remain — clean state
```

---

## 6. Interpretability Pipeline

```mermaid
graph LR
    CKP["Trained checkpoint
    checkpoints/small/best.pt"]

    CKP --> AV["AttentionVisualizer
    What does each head attend to?"]

    CKP --> LL["LogitLens
    What does each layer predict?"]

    CKP --> LP["LinearProbe
    What properties are encoded?"]

    CKP --> AP["ActivationPatcher
    Which components are causal?"]

    CKP --> IH["InductionHeadDetector
    Which heads do induction?"]

    CKP --> CA["CircuitAblator
    What is the minimal circuit?"]

    AV --> HYP["Hypothesis formation"]
    LL --> HYP
    LP --> HYP

    HYP --> AP
    AP --> IH
    IH --> CA

    CA --> FND["Finding:
    circuit description +
    causal evidence"]

    FND --> PAP["Paper / report
    research/reports/"]

    style FND fill:#1a3a1a,stroke:#4ac44a,color:#fff
    style CKP fill:#3a2a1a,stroke:#c4a44a,color:#fff
```

---

## 7. Training Pipeline

```mermaid
graph TD
    RAW["Raw text corpus
    data/tinystories.txt"]

    RAW --> BPE["BPETokenizer.train()
    vocab_size=8192"]

    BPE --> TKN["Tokenizer saved
    checkpoints/small/tokenizer.json"]

    RAW --> PRE["tokenize_corpus.py
    → data/tinystories_tokens.bin"]

    TKN --> PRE

    PRE --> DL["DataLoader
    sequences of length context_length"]

    DL --> TR["Trainer.train()
    ─────────────────
    forward → loss
    backward → grads
    clip → optimiser step
    LR schedule
    checkpoint every N steps"]

    TR --> CKP["checkpoints/small/
    step_*.pt
    best.pt
    last.pt"]

    CKP --> EVL["evaluate.py
    val perplexity
    text generation"]

    style TR fill:#1a2a3a,stroke:#4a8ac4,color:#fff
```
