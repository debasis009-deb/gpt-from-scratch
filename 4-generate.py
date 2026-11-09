"""Generate text from the saved checkpoint - no data download or training needed.

Run:  python generate.py "First Citizen:"
Options: --chars 400 --temperature 0.8 --top-k 50
Lower temperature / smaller top-k = safer, more repetitive text.
"""

import argparse
import os
import sys

import torch

from model import GPT


def load_artifact(path='checkpoints/my-gpt-model.pt', device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    artifact = torch.load(path, map_location=device)
    config = artifact['config']
    model = GPT(config['vocab_size'], config['n_embd'], config['n_head'],
                config['n_layer'], config['block_size'], config['dropout']).to(device)
    model.load_state_dict(artifact['model_state'])
    model.eval()
    stoi, itos = artifact['stoi'], artifact['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join(itos[i] for i in l)
    return model, encode, decode, config['block_size'], device, artifact


@torch.no_grad()
def sample(model, encode, decode, block_size, device, prompt,
           max_chars=400, temperature=0.8, top_k=50):
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    for _ in range(max_chars):
        logits, _ = model(idx[:, -block_size:])
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf')
        probs = torch.softmax(logits, dim=-1)
        idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
    return decode(idx[0].tolist())


def positive_int(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError(f'must be a positive integer, got {value!r}')
    return n


def positive_float(value):
    f = float(value)
    if f <= 0:
        raise argparse.ArgumentTypeError(f'must be positive, got {value!r}')
    return f


def main():
    parser = argparse.ArgumentParser(description='Generate Shakespeare-style text.')
    parser.add_argument('prompt', nargs='?', default='\nFirst Citizen:\n')
    parser.add_argument('--chars', type=positive_int, default=400)
    parser.add_argument('--temperature', type=positive_float, default=0.8)
    parser.add_argument('--top-k', type=positive_int, default=50)
    parser.add_argument('--checkpoint', default='checkpoints/my-gpt-model.pt')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        sys.exit(f"checkpoint not found: {args.checkpoint} - run `python train.py` first, "
                 "or place a previously saved copy at that path")
    model, encode, decode, block_size, device, artifact = load_artifact(args.checkpoint)

    vocab_size = len(artifact['stoi'])
    if not args.prompt:
        sys.exit('empty prompt - pass some starting text, e.g. python generate.py "First Citizen:"')
    unknown = sorted(set(args.prompt) - set(artifact['stoi']))
    if unknown:
        sys.exit(f"prompt contains characters outside the {vocab_size}-character "
                 f"vocabulary: {''.join(unknown)!r}")
    if args.top_k > vocab_size:
        sys.exit(f'--top-k {args.top_k} exceeds the vocabulary size ({vocab_size})')

    print('loaded checkpoint:', artifact['steps'], 'training steps,'
          ' val loss', round(artifact['val_loss'], 4))
    print(sample(model, encode, decode, block_size, device,
                 args.prompt, args.chars, args.temperature, args.top_k))


if __name__ == '__main__':
    main()
