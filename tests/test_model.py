"""Fast CPU tests: tokenizer round-trip, model shapes, generation, artifact I/O."""

import torch

from model import GPT
from train import build_tokenizer, get_batch


def tiny_model():
    return GPT(vocab_size=32, n_embd=32, n_head=2, n_layer=2,
               block_size=16, dropout=0.0)


def test_tokenizer_round_trip():
    text = 'To be, or not to be: that is the question.'
    stoi, itos, encode, decode, vocab_size = build_tokenizer(text)
    assert decode(encode(text)) == text
    assert vocab_size == len(set(text))
    assert all(0 <= i < vocab_size for i in encode(text))


def test_forward_shapes_and_loss():
    model = tiny_model()
    idx = torch.randint(0, 32, (4, 16))
    logits, loss = model(idx)
    assert logits.shape == (4, 16, 32)
    assert loss is None
    logits, loss = model(idx, idx)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_generate_length():
    model = tiny_model()
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=10)
    assert out.shape == (1, 11)


def test_get_batch_shapes_and_shift():
    data = torch.arange(100)
    torch.manual_seed(0)
    x, y = get_batch(data, batch_size=8, block_size=16, device='cpu')
    assert x.shape == (8, 16) and y.shape == (8, 16)
    assert torch.equal(y, x + 1)  # targets are inputs shifted by one character


def test_training_reduces_loss():
    torch.manual_seed(0)
    model = tiny_model()
    data = torch.arange(2000) % 32  # repeating pattern the model can learn
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first_loss = last_loss = None
    for step in range(40):
        xb, yb = get_batch(data, 8, 16, 'cpu')
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0:
            first_loss = loss.item()
        last_loss = loss.item()
    assert last_loss < first_loss


def test_artifact_round_trip(tmp_path):
    model = tiny_model()
    artifact = {
        'model_state': model.state_dict(),
        'config': {'n_embd': 32, 'n_head': 2, 'n_layer': 2,
                   'block_size': 16, 'vocab_size': 32, 'dropout': 0.0},
        'stoi': {'a': 0},
        'itos': {0: 'a'},
        'steps': 1,
        'val_loss': 1.0,
    }
    path = tmp_path / 'artifact.pt'
    torch.save(artifact, path)
    loaded = torch.load(path)
    restored = tiny_model()
    restored.load_state_dict(loaded['model_state'])
    assert loaded['config']['block_size'] == 16
    assert set(loaded) == {'model_state', 'config', 'stoi', 'itos', 'steps', 'val_loss'}
