"""End-to-end CLI tests: tiny CPU runs of the real train/eval/generate entrypoints.

These run the actual scripts as subprocesses with a tiny model config, so the
full training loop, checkpointing, resume, artifact packing, eval-only and
generation paths are all exercised without needing a GPU or much memory.
"""

import os
import subprocess
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TINY = ['--n-embd', '32', '--n-head', '2', '--n-layer', '2', '--block-size', '32',
        '--batch-size', '8', '--eval-iters', '2']


def run(script, *cli, cwd):
    return subprocess.run([sys.executable, os.path.join(REPO, script), *cli],
                          cwd=cwd, capture_output=True, text=True, timeout=300)


@pytest.fixture()
def workdir(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    (data_dir / 'input.txt').write_text('to be or not to be, that is the question.\n' * 500)
    return tmp_path


def train(workdir, *extra):
    return run('train.py', '--data-dir', str(workdir / 'data'), *TINY, *extra, cwd=workdir)


def test_train_then_eval_only_and_generate(workdir):
    result = train(workdir, '--max-iters', '4', '--eval-interval', '2')
    assert result.returncode == 0, result.stderr
    artifact = workdir / 'checkpoints' / 'my-gpt-model.pt'
    assert artifact.exists()
    assert (workdir / 'checkpoints' / 'model.pt').exists()
    assert (workdir / 'checkpoints' / 'train_state.pt').exists()
    assert 'step 4:' in result.stdout  # final post-step weights were evaluated

    result = run('train.py', '--data-dir', str(workdir / 'data'),
                 '--eval-only', '--checkpoint', str(artifact), '--eval-iters', '2', cwd=workdir)
    assert result.returncode == 0, result.stderr
    assert 'val ' in result.stdout

    result = run('generate.py', 'to be', '--chars', '20', '--top-k', '10',
                 '--checkpoint', str(artifact), cwd=workdir)
    assert result.returncode == 0, result.stderr
    assert 'loaded checkpoint:' in result.stdout


def load_artifact(path):
    return torch.load(path, weights_only=False)


def clone_data(workdir, other):
    (other / 'data').mkdir(parents=True)
    (other / 'data' / 'input.txt').write_text((workdir / 'data' / 'input.txt').read_text())


def assert_same_state(a_path, b_path):
    a = load_artifact(a_path)
    b = load_artifact(b_path)
    assert a['step'] == b['step']
    assert a['best_val'] == b['best_val']
    assert a['best_step'] == b['best_step']
    assert a['no_improve'] == b['no_improve']
    assert torch.equal(a['rng_state'], b['rng_state'])
    for k in a['model_state']:
        assert torch.equal(a['model_state'][k], b['model_state'][k]), k
    assert a['optimizer_state']['state'].keys() == b['optimizer_state']['state'].keys()
    for p in a['optimizer_state']['state']:
        for key, va in a['optimizer_state']['state'][p].items():
            vb = b['optimizer_state']['state'][p][key]
            if torch.is_tensor(va):
                assert torch.equal(va, vb), (p, key)
            else:
                assert va == vb, (p, key)


def assert_same_artifact(a_path, b_path):
    a = load_artifact(a_path)
    b = load_artifact(b_path)
    assert a['steps'] == b['steps']
    assert a['val_loss'] == b['val_loss']
    for k in a['model_state']:
        assert torch.equal(a['model_state'][k], b['model_state'][k]), k


def test_resume_boundary_split_matches_uninterrupted(workdir, tmp_path):
    # uninterrupted 8-step run vs a 4 + resumed-4 split (4 lands on eval boundary)
    result = train(workdir, '--max-iters', '8', '--eval-interval', '2')
    assert result.returncode == 0, result.stderr

    other = tmp_path / 'split'
    clone_data(workdir, other)
    result = train(other, '--max-iters', '4', '--eval-interval', '2')
    assert result.returncode == 0, result.stderr
    result = train(other, '--max-iters', '8', '--eval-interval', '2', '--resume')
    assert result.returncode == 0, result.stderr
    assert 'resumed from' in result.stdout
    assert 'step 8:' in result.stdout  # reached the new max, did not restart

    assert_same_artifact(workdir / 'checkpoints' / 'my-gpt-model.pt',
                         other / 'checkpoints' / 'my-gpt-model.pt')
    assert_same_state(workdir / 'checkpoints' / 'train_state.pt',
                      other / 'checkpoints' / 'train_state.pt')


def test_resume_off_boundary_split_keeps_identical_weights(workdir, tmp_path):
    # 3 + resumed-5 split: leg 1 ends between eval boundaries
    result = train(workdir, '--max-iters', '8', '--eval-interval', '2')
    assert result.returncode == 0, result.stderr

    other = tmp_path / 'split'
    clone_data(workdir, other)
    result = train(other, '--max-iters', '3', '--eval-interval', '2')
    assert result.returncode == 0, result.stderr
    result = train(other, '--max-iters', '8', '--eval-interval', '2', '--resume')
    assert result.returncode == 0, result.stderr

    # weights, optimizer and RNG are bitwise identical; only the best-tracker
    # may differ by the one extra evaluation a mid-boundary leg end performs
    assert_same_state(workdir / 'checkpoints' / 'train_state.pt',
                      other / 'checkpoints' / 'train_state.pt')


def test_resume_after_early_stop_refuses(workdir, tmp_path):
    # learning rate 0: validation never improves, both runs stop early at step 6
    result = train(workdir, '--max-iters', '8', '--eval-interval', '2',
                   '--learning-rate', '0')
    assert result.returncode == 0, result.stderr
    assert 'stopping early' in result.stdout

    other = tmp_path / 'split'
    clone_data(workdir, other)
    result = train(other, '--max-iters', '4', '--eval-interval', '2', '--learning-rate', '0')
    assert result.returncode == 0, result.stderr
    result = train(other, '--max-iters', '8', '--eval-interval', '2', '--resume')
    assert result.returncode == 0, result.stderr
    assert 'stopping early' in result.stdout

    assert_same_artifact(workdir / 'checkpoints' / 'my-gpt-model.pt',
                         other / 'checkpoints' / 'my-gpt-model.pt')

    # resuming an early-stopped run refuses and changes nothing
    before = load_artifact(workdir / 'checkpoints' / 'my-gpt-model.pt')
    result = train(workdir, '--max-iters', '12', '--eval-interval', '2', '--resume')
    assert result.returncode != 0
    assert 'stopped early' in result.stderr + result.stdout
    after = load_artifact(workdir / 'checkpoints' / 'my-gpt-model.pt')
    assert after['steps'] == before['steps']
    for k in before['model_state']:
        assert torch.equal(before['model_state'][k], after['model_state'][k]), k


def test_resume_rejects_stale_or_mismatched_state(workdir):
    result = train(workdir, '--max-iters', '4', '--eval-interval', '2')
    assert result.returncode == 0, result.stderr

    # max-iters at or below the saved step is rejected, artifact untouched
    result = train(workdir, '--max-iters', '4', '--eval-interval', '2', '--resume')
    assert result.returncode != 0
    assert 'larger --max-iters' in result.stderr + result.stdout

    # changing the training text is caught by the data fingerprint
    # same vocabulary (so config/tokenizer checks pass) but different bytes
    with open(workdir / 'data' / 'input.txt', 'a') as f:
        f.write('or to be or to be, that is not the question.\n')
    result = train(workdir, '--max-iters', '6', '--eval-interval', '2', '--resume')
    assert result.returncode != 0
    assert 'fingerprint' in result.stderr + result.stdout

    # changing the model shape is caught by the config check
    result = train(workdir, '--max-iters', '6', '--eval-interval', '2', '--resume',
                   '--n-embd', '64', '--n-head', '2', '--n-layer', '2', '--block-size', '32')
    assert result.returncode != 0
    assert 'config mismatch' in result.stderr + result.stdout


def test_worse_resume_does_not_clobber_best_artifact(workdir):
    result = train(workdir, '--max-iters', '4', '--eval-interval', '2')
    assert result.returncode == 0, result.stderr
    before = load_artifact(workdir / 'checkpoints' / 'my-gpt-model.pt')

    # resume with an absurd learning rate: training diverges to NaN, best must survive
    result = train(workdir, '--max-iters', '6', '--eval-interval', '2', '--resume',
                   '--learning-rate', '1000000000.0')
    assert result.returncode == 0, result.stderr
    after = load_artifact(workdir / 'checkpoints' / 'my-gpt-model.pt')
    assert after['steps'] == before['steps']
    assert after['val_loss'] == before['val_loss']
    for k in before['model_state']:
        assert torch.equal(before['model_state'][k], after['model_state'][k]), k


def test_train_rejects_bad_arguments(workdir):
    result = train(workdir, '--max-iters', '0')
    assert result.returncode != 0
    result = train(workdir, '--max-iters', '2', '--n-embd', '30', '--n-head', '4')
    assert result.returncode != 0  # 30 not divisible by 4
    result = train(workdir, '--max-iters', '2', '--data-dir', str(workdir / 'missing'))
    assert result.returncode != 0  # no input.txt in a custom data dir


def test_generate_rejects_bad_input(workdir):
    result = train(workdir, '--max-iters', '2', '--eval-interval', '1')
    assert result.returncode == 0, result.stderr
    artifact = str(workdir / 'checkpoints' / 'my-gpt-model.pt')

    result = run('generate.py', 'to be', '--chars', '5', '--top-k', '99999',
                 '--checkpoint', artifact, cwd=workdir)
    assert result.returncode != 0
    assert 'vocabulary' in result.stderr + result.stdout

    result = run('generate.py', 'to be \u20ac', '--chars', '5',
                 '--checkpoint', artifact, cwd=workdir)
    assert result.returncode != 0  # euro sign is not in the vocabulary

    result = run('generate.py', '', '--chars', '5', '--checkpoint', artifact, cwd=workdir)
    assert result.returncode != 0  # empty prompt

    result = run('generate.py', 'to be', '--chars', '0', '--checkpoint', artifact, cwd=workdir)
    assert result.returncode != 0  # non-positive --chars
