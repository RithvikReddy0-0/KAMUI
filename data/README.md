# Data

Raw data files are never committed to this repository.
Download them using the scripts below.

## Datasets

### TinyShakespeare (1MB)
Used for: nano config. Expect about a minute of tokenizer work on a laptop CPU
before training starts.
```bash
curl -L -o data/tinyshakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### TinyStories (2GB)
Used for: small config. It's far larger than KAMUI's pure-Python BPE tokenizer can
comfortably handle, so train on a slice of it.
```bash
# Download from HuggingFace
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('roneneldan/TinyStories', split='train')
with open('data/tinystories.txt', 'w', encoding='utf-8') as f:
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
with open('data/openwebtext10k.txt', 'w', encoding='utf-8') as f:
    for item in ds:
        f.write(item['text'] + '\n<|endoftext|>\n')
"
```

## After downloading

Train directly from the text file. `kamui-train` trains its own tokenizer from the
corpus each time, and writes it next to the checkpoint:
```bash
python scripts/train.py --config configs/nano.yaml --corpus data/tinyshakespeare.txt
```

`scripts/tokenize_corpus.py` saves a tokenizer and a `.npy` token array for your own
scripts. `kamui-train` can't read that array yet, so it won't speed up training:
```bash
python scripts/tokenize_corpus.py \
    --input data/tinyshakespeare.txt \
    --vocab-size 4096 \
    --tokenizer-out data/tinyshakespeare_tokenizer.json \
    --tokens-out data/tinyshakespeare_tokens
```
