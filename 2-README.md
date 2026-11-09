# GPT from scratch

A small GPT (transformer) written and trained from scratch in PyTorch on the
TinyShakespeare corpus (~1 MB of Shakespeare plays). No pretrained weights, no
high-level trainer libraries - the tokenizer, model, training loop and text
generation are all plain PyTorch in this repo.

## Results

All numbers below are from one continuous training run (2,000 steps, then
3,000 more):

| Training steps | Held-out validation loss (lower is better) |
| -------------- | ------------------------------------------ |
| 2,000          | 1.5900                                     |
| 4,750          | 1.4729 (best, saved)                       |
| 5,000          | 1.4865 (final)                             |

The saved checkpoint is the best one from this run: validation loss 1.4729 at
step 4,750. Training then continued to step 5,000, where validation loss sat
at 1.4865 - the better step-4,750 weights are what get saved.

Training saves one complete checkpoint (`checkpoints/my-gpt-model.pt`, ~53 MB)
with the weights, model config and character tokenizer in a single file. The
checkpoint is not committed to this repo (large binary, git-ignored) - run
`python train.py` to produce it yourself, or use a previously saved copy.

## What it does

Generates Shakespeare-style dialogue one character at a time:

```
First Citizen:
Therefore cannot perceive the drink, but that thou
Shalt be cannot need, to take death.

CORIOLANUS:
Art thou ship at all and contents, but think
Prepared by so traitors with the false slaves,
```

(actual output of `python generate.py` on the saved checkpoint)

## Limitations

This is a small character-level model trained on 1 MB of one author's writing.
Its output still contains wrong words and broken sentences. It is a learning
project that shows how an LLM works from the inside, not a chatbot.

## Layout

- `model.py` - the transformer: attention head, multi-head attention,
  feedforward, block, and the GPT itself
- `train.py` - downloads the data, builds the tokenizer, trains, keeps the
  best checkpoint and packs the complete artifact
- `generate.py` - loads the artifact and generates text (no training needed)
- `tests/test_model.py` - tokenizer, model-shape, generation and artifact tests

## Run it

```bash
pip install -r requirements.txt

# train from scratch (GPU recommended, ~1 hour on a free Colab T4)
python train.py

# common options (everything is a flag - see python train.py --help)
python train.py --max-iters 5000 --eval-interval 250
python train.py --resume       # continue from checkpoints/train_state.pt with full
                               # state (weights, optimizer, step, best tracker, RNG).
                               # On CPU a stop-and-resume replays the uninterrupted
                               # run bitwise; a leg ending between eval boundaries
                               # adds one extra best-tracker event. Early-stopped
                               # runs refuse to resume. GPU resume is untested.
python train.py --eval-only    # report the saved artifact's losses, no training

# generate text from the saved checkpoint
python generate.py "First Citizen:" --chars 400 --temperature 0.8 --top-k 50

# run the tests
python -m pytest
```

Data (TinyShakespeare) downloads automatically on the first training run, and
its SHA-256 checksum is verified before use. Training fixes the Python-side
random seed (`torch.manual_seed(1337)`); GPU kernel scheduling can still vary,
so exact bit-identical reruns are not guaranteed across different GPUs or
driver versions. The numbers above are from the author's run on a free Colab T4.

Verified environments: the recorded training run used a free Colab T4 with its
preinstalled PyTorch; the test suite and inference were re-verified on Linux
with torch 2.14.1 (CPU build). A from-scratch OS-level install has not been
separately tested - `pip install -r requirements.txt` is the expected setup.

