"""Train the GPT on TinyShakespeare and save one complete checkpoint.

Run:  python train.py           (all knobs are flags - see python train.py --help)
Data downloads itself on the first run and its checksum is verified. The best
weights (lowest validation loss) are kept in checkpoints/model.pt, full resume
state in checkpoints/train_state.pt, and the final artifact (weights + config
+ tokenizer) is packed into checkpoints/my-gpt-model.pt.

Resume semantics (--resume), verified on CPU:
- Evaluation uses its own fixed-seed generator, so evaluating never perturbs
  training randomness. A run stopped and resumed mid-way continues with
  bitwise-identical weights, optimizer state, RNG and best-tracker when the
  previous leg ended on an evaluation boundary; a leg ending between
  boundaries adds one extra best-tracker event, which can shift early-stop
  timing by one evaluation.
- A run that stopped early refuses to resume (the plateau would stop it
  again); delete checkpoints/train_state.pt to start over.
- CPU resume is tested; CUDA RNG state is saved/restored, but GPU resume and
  cross-device resume rely on PyTorch state_dict device migration - untested.
"""

import argparse
import hashlib
import os
import sys
import urllib.request

import torch

from model import GPT

# default hyperparameters (the recorded run used exactly these)
BATCH_SIZE = 64      # sequences trained in parallel
BLOCK_SIZE = 256     # context length
MAX_ITERS = 5000
EVAL_INTERVAL = 250
EVAL_ITERS = 50
LEARNING_RATE = 3e-4
N_EMBD = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2

EVAL_SEED = 4242     # fixed seed for evaluation batches: same eval every time

DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
DATA_SHA256 = '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'
DATA_DIR = 'data/raw'
CHECKPOINT_DIR = 'checkpoints'


def build_tokenizer(text):
    # character <-> number
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join(itos[i] for i in l)
    return stoi, itos, encode, decode, len(chars)


def get_batch(data, batch_size, block_size, device, generator=None):
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, device, batch_size, block_size, eval_iters=EVAL_ITERS):
    # dedicated generator: evaluation never consumes training randomness, and
    # every evaluation of the same weights sees the same batches
    generator = torch.Generator().manual_seed(EVAL_SEED)
    out = {}
    model.eval()
    for split, d in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(d, batch_size, block_size, device, generator)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


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


def nonnegative_float(value):
    f = float(value)
    if f < 0:
        raise argparse.ArgumentTypeError(f'must be non-negative, got {value!r}')
    return f


def load_data(data_dir):
    """Read data_dir/input.txt, downloading TinyShakespeare for the default dir only.

    Returns (text, sha256 of the raw file). The pinned SHA-256 check applies to
    the canonical TinyShakespeare file; a custom --data-dir is the caller's own
    data - its fingerprint is only used to validate resumes.
    """
    data_path = os.path.join(data_dir, 'input.txt')
    if not os.path.exists(data_path):
        if data_dir != DATA_DIR:
            sys.exit(f'no input.txt in {data_dir} - place your training text there')
        print('downloading TinyShakespeare...')
        os.makedirs(data_dir, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, data_path)
    with open(data_path, 'rb') as f:
        raw = f.read()
    digest = hashlib.sha256(raw).hexdigest()
    if data_dir == DATA_DIR and digest != DATA_SHA256:
        sys.exit(f'dataset checksum mismatch: {data_path} - delete it and rerun to re-download')
    return raw.decode('utf-8'), digest


def parse_args():
    parser = argparse.ArgumentParser(description='Train the GPT on TinyShakespeare.')
    parser.add_argument('--max-iters', type=positive_int, default=MAX_ITERS)
    parser.add_argument('--eval-interval', type=positive_int, default=EVAL_INTERVAL)
    parser.add_argument('--eval-iters', type=positive_int, default=EVAL_ITERS,
                        help='batches averaged per loss estimate (lower = faster, noisier)')
    parser.add_argument('--batch-size', type=positive_int, default=BATCH_SIZE)
    parser.add_argument('--block-size', type=positive_int, default=BLOCK_SIZE)
    parser.add_argument('--n-embd', type=positive_int, default=N_EMBD)
    parser.add_argument('--n-head', type=positive_int, default=N_HEAD)
    parser.add_argument('--n-layer', type=positive_int, default=N_LAYER)
    parser.add_argument('--dropout', type=float, default=DROPOUT)
    parser.add_argument('--learning-rate', type=nonnegative_float, default=None,
                        help=f'default {LEARNING_RATE}; on --resume it overrides the saved one')
    parser.add_argument('--data-dir', default=DATA_DIR,
                        help='folder containing input.txt (default downloads TinyShakespeare)')
    parser.add_argument('--resume', action='store_true',
                        help='continue from checkpoints/train_state.pt (tested on CPU; '
                             'early-stopped runs refuse to resume)')
    parser.add_argument('--eval-only', action='store_true',
                        help='skip training: load the saved artifact and report its losses')
    parser.add_argument('--checkpoint', default=os.path.join(CHECKPOINT_DIR, 'my-gpt-model.pt'),
                        help='artifact path used by --eval-only')
    args = parser.parse_args()
    if not 0.0 <= args.dropout < 1.0:
        parser.error(f'--dropout must be in [0, 1), got {args.dropout}')
    if args.n_embd % args.n_head != 0:
        parser.error(f'--n-embd ({args.n_embd}) must be divisible by --n-head ({args.n_head})')
    return args


def main():
    args = parse_args()

    torch.manual_seed(1337)  # fixes Python-side randomness; GPU kernels can still vary slightly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    text, data_digest = load_data(args.data_dir)

    if args.eval_only:
        # evaluate exactly what was saved: the artifact's own config and tokenizer
        artifact = torch.load(args.checkpoint, map_location=device)
        config = artifact['config']
        model = GPT(config['vocab_size'], config['n_embd'], config['n_head'],
                    config['n_layer'], config['block_size'], config['dropout']).to(device)
        model.load_state_dict(artifact['model_state'])
        stoi = artifact['stoi']
        data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        n = int(0.9 * len(data))
        losses = estimate_loss(model, data[:n], data[n:], device,
                               args.batch_size, config['block_size'], args.eval_iters)
        print(f"artifact {args.checkpoint}: train {losses['train']:.4f}, val {losses['val']:.4f}")
        return

    stoi, itos, encode, decode, vocab_size = build_tokenizer(text)
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    config = {'n_embd': args.n_embd, 'n_head': args.n_head, 'n_layer': args.n_layer,
              'block_size': args.block_size, 'vocab_size': vocab_size,
              'dropout': args.dropout}

    model = GPT(vocab_size, args.n_embd, args.n_head, args.n_layer,
                args.block_size, args.dropout).to(device)
    print(sum(p.numel() for p in model.parameters()) / 1e6, 'million parameters')
    lr = LEARNING_RATE if args.learning_rate is None else args.learning_rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val = float('inf')
    best_step = 0
    no_improve = 0
    start_step = 0
    last_evaluated_step = -1
    model_path = os.path.join(CHECKPOINT_DIR, 'model.pt')
    state_path = os.path.join(CHECKPOINT_DIR, 'train_state.pt')

    if args.resume:
        # loaded on CPU so the saved RNG state stays a CPU tensor
        state = torch.load(state_path, map_location='cpu')
        mismatches = [k for k, v in config.items() if state['config'].get(k) != v]
        if mismatches:
            sys.exit(f'config mismatch with {state_path}: {mismatches} - '
                     'resume with the same model/data settings')
        if state['stoi'] != stoi:
            sys.exit(f'tokenizer mismatch with {state_path} - '
                     'the training text does not produce the saved vocabulary')
        if state['data_sha256'] != data_digest:
            sys.exit('data fingerprint mismatch: input.txt changed since the saved state')
        if state['stopped_early']:
            sys.exit(f"this run already stopped early at step {state['step']} "
                     f"(val plateau; best is step {state['best_step']}) - "
                     'delete checkpoints/train_state.pt to start over')
        if args.max_iters <= state['step']:
            sys.exit(f'saved state is already at step {state["step"]} - '
                     'pass a larger --max-iters to continue training')
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        if args.learning_rate is not None:  # explicit flag wins over the saved lr
            for group in optimizer.param_groups:
                group['lr'] = args.learning_rate
        start_step = state['step']
        best_val = state['best_val']
        best_step = state['best_step']
        no_improve = state['no_improve']
        last_evaluated_step = state['last_evaluated_step']
        torch.set_rng_state(state['rng_state'])
        if device == 'cuda' and state.get('cuda_rng_state_all') is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng_state_all'])
        print(f"resumed from {state_path} at step {start_step}"
              f" (best so far: val {best_val:.4f} at step {best_step})")

    def evaluate(step):
        # report, keep best weights, feed the early-stop counter
        nonlocal best_val, best_step, no_improve, last_evaluated_step
        losses = estimate_loss(model, train_data, val_data, device,
                               args.batch_size, args.block_size, args.eval_iters)
        print(f"step {step}: train {losses['train']:.4f}, val {losses['val']:.4f}")
        last_evaluated_step = step
        if losses['val'] < best_val:
            best_val = losses['val']
            best_step = step
            torch.save(model.state_dict(), model_path)
            no_improve = 0
        else:
            no_improve += 1

    def save_state(step, stopped_early=False):
        # snapshot of the full training state at `step` (= updates completed).
        # Evaluation consumes no global RNG, so this RNG state is exactly the
        # one an uninterrupted run has after the same number of updates.
        torch.save({'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'step': step,
                    'best_val': best_val,
                    'best_step': best_step,
                    'no_improve': no_improve,
                    'last_evaluated_step': last_evaluated_step,
                    'stopped_early': stopped_early,
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state_all': (torch.cuda.get_rng_state_all()
                                           if device == 'cuda' else None),
                    'config': config,
                    'stoi': stoi,
                    'data_sha256': data_digest}, state_path)

    stopped_early = False
    for step in range(start_step, args.max_iters):
        if step % args.eval_interval == 0 and step != last_evaluated_step:
            evaluate(step)
            save_state(step)  # pre-update snapshot: resume redoes no work
            if no_improve >= 3:
                print('val loss stopped improving, stopping early')
                save_state(step, stopped_early=True)
                stopped_early = True
                break

        xb, yb = get_batch(train_data, args.batch_size, args.block_size, device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if not stopped_early:
        # the weights after the final update count too - evaluate them as well
        evaluate(args.max_iters)
        save_state(args.max_iters)

    # save one complete artifact: best weights + config + tokenizer in a single file
    artifact = {
        'model_state': torch.load(model_path),
        'config': config,
        'stoi': stoi,
        'itos': itos,
        'steps': best_step,
        'val_loss': best_val,
    }
    artifact_path = os.path.join(CHECKPOINT_DIR, 'my-gpt-model.pt')
    torch.save(artifact, artifact_path)
    print('artifact saved:', round(os.path.getsize(artifact_path) / 1e6, 1), 'MB')


if __name__ == '__main__':
    main()
