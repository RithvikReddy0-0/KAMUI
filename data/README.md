# Data

Raw data files are never committed to this repository.
Download them using the scripts below.

## Datasets

### TinyShakespeare (1MB)
Used for: nano config, quick debugging, CI integration tests
```bash
wget -O data/tinystories.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### TinyStories (2GB)
Used for: small config, primary interpretability experiments
```bash
# Download from HuggingFace
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('roneneldan/TinyStories', split='train')
with open('data/tinystories.txt', 'w') as f:
    for item in ds:
        f.write(item['text'] + '\n<|endoftext|>\n')
"
```

### OpenWebText-10k (subset)
Used for: medium config, GPT-2 comparison experiments
```bash
python -c "
from datasets import load_dataset
ds = load_dataset('Elriggs/openwebtext-100k', split='train[:10000]')
with open('data/openwebtext10k.txt', 'w') as f:
    for item in ds:
        f.write(item['text'] + '\n<|endoftext|>\n')
"
```

## After downloading

Pre-tokenise for fast training:
```bash
python scripts/tokenize_corpus.py \
    --input data/tinystories.txt \
    --tokenizer checkpoints/tokenizer.json \
    --output data/tinystories_tokens.bin
```
