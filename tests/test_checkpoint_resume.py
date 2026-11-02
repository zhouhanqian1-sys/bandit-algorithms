"""Verify round checkpoint recovery with tiny test-only rewards and no LLM calls."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

from validate_notebook import load_definitions, read_notebook


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def make_namespace(notebook):
    namespace = load_definitions(notebook, np)
    namespace.update(
        movies=[('First', 'Test movie one'), ('Second', 'Test movie two'),
                ('Third', 'Test movie three')],
        roles={'A': 'Test profile'}, movie_noise_std=0.1,
        medical_profiles={'A': 'Test patient'},
        medical_questions=[
            {'prompt_id': 'q0', 'prompt': 'Test question zero',
             'answers': [{'text': 'First'}, {'text': 'Second'}]},
            {'prompt_id': 'q1', 'prompt': 'Test question one',
             'answers': [{'text': 'First'}, {'text': 'Second'}, {'text': 'Third'}]},
        ],
        medical_features=[np.eye(3)[:2], np.eye(3)],
    )
    return namespace


def check_movie_resume(notebook, directory):
    namespace = make_namespace(notebook)
    calls = []

    def reward(arm, role):
        calls.append(arm)
        return float(np.random.random())  # The runner supplies a fresh deterministic noise seed.

    namespace['movie_reward'] = reward
    run = namespace['run_movie_bandits']
    expected = run('A', N=2, T=7, seed=91)
    path = directory / 'movie.json'
    calls.clear()

    def interrupted_reward(arm, role):
        if len(calls) == 8:
            raise RuntimeError('Test-only interruption during a round.')
        return reward(arm, role)

    namespace['movie_reward'] = interrupted_reward
    try:
        run('A', N=2, T=7, seed=91, checkpoint_path=path, checkpoint_every=2)
        raise AssertionError('The test interruption did not occur.')
    except RuntimeError:
        pass
    assert path.exists()
    saved = json.loads(path.read_text())
    assert saved['next_run'] < 2
    assert 'round_rewards' not in saved
    namespace['movie_reward'] = reward
    actual = run('A', N=2, T=7, seed=91, checkpoint_path=path, checkpoint_every=2)
    assert all(np.array_equal(a, b) for a, b in zip(actual, expected))
    call_count = len(calls)
    complete = run('A', N=2, T=7, seed=91, checkpoint_path=path)
    assert len(calls) == call_count, 'A complete checkpoint must not call the reward again.'
    assert all(np.array_equal(a, b) for a, b in zip(complete, expected))
    try:
        run('A', N=2, T=7, seed=92, checkpoint_path=path)
        raise AssertionError('An incompatible seed must reject an old checkpoint.')
    except ValueError:
        pass


def check_medical_resume(notebook, directory):
    namespace = make_namespace(notebook)
    calls = []

    def reward(question, profile, arm, seed):
        calls.append((question['prompt_id'], arm, seed))
        return float(np.random.default_rng(seed).random())

    namespace['patient_reward'] = reward
    run = namespace['run_medical_bandits']
    expected = run('A', N=2, T=7, seed=93)
    path = directory / 'medical.json'
    calls.clear()

    def interrupted_reward(question, profile, arm, seed):
        if len(calls) == 7:
            raise RuntimeError('Test-only interruption during a round.')
        return reward(question, profile, arm, seed)

    namespace['patient_reward'] = interrupted_reward
    try:
        run('A', N=2, T=7, seed=93, checkpoint_path=path, checkpoint_every=2)
        raise AssertionError('The test interruption did not occur.')
    except RuntimeError:
        pass
    assert path.exists()
    saved = json.loads(path.read_text())
    assert saved['next_run'] < 2
    assert 'round_rewards' not in saved
    namespace['patient_reward'] = reward
    actual = run('A', N=2, T=7, seed=93, checkpoint_path=path, checkpoint_every=2)
    assert all(np.array_equal(a, b) for a, b in zip(actual, expected))
    call_count = len(calls)
    complete = run('A', N=2, T=7, seed=93, checkpoint_path=path)
    assert len(calls) == call_count
    assert all(np.array_equal(a, b) for a, b in zip(complete, expected))
    namespace['medical_profiles']['A'] = 'Different preferences'
    try:
        run('A', N=2, T=7, seed=93, checkpoint_path=path)
        raise AssertionError('Changed experiment inputs must reject an old checkpoint.')
    except ValueError:
        pass


def main():
    # Test-only rewards must never touch the production checkpoint directory.
    os.environ.pop('BIOSTAT825_CHECKPOINT_DIR', None)
    notebook = read_notebook(PROJECT / 'Experiments.ipynb')
    with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
        directory = Path(temporary)
        check_movie_resume(notebook, directory)
        check_medical_resume(notebook, directory)
    print('PASS movie and medical interrupted-round recovery exactly matches uninterrupted runs.')
    print('PASS completed checkpoints avoid reward replay; metadata rejects incompatible settings.')
    print('Validation used test-only rewards; no LLM model or dataset was loaded.')


if __name__ == '__main__':
    main()
