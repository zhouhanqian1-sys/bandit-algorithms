"""Small JSON checkpoints for the notebook's NumPy-based LLM bandit policies."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


STATE_FIELDS = ('t', 'counts', 'mean_rewards', 'cov', 'invcov', 'b_t', 'hat_theta')


def json_value(value):
    """Convert NumPy state to JSON without serializing executable Python objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Cannot store checkpoint value of type {type(value).__name__}.')


def checkpoint_signature(values):
    encoded = json.dumps(values, sort_keys=True, default=json_value, allow_nan=False)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def capture_agent_states(agents):
    """Store learned estimates and each policy's independent random-generator state."""
    states = []
    for agent in agents:
        state = {'name': agent.name(), 'rng_state': agent.rng.bit_generator.state}
        for field in STATE_FIELDS:
            if hasattr(agent, field):
                state[field] = getattr(agent, field)
        states.append(state)
    return states


def restore_agent_states(agents, states):
    if len(agents) != len(states):
        raise ValueError('The checkpoint has a different number of algorithms.')
    for agent, state in zip(agents, states):
        if agent.name() != state['name']:
            raise ValueError('The checkpoint algorithm configuration has changed.')
        agent.rng.bit_generator.state = state['rng_state']
        for field in STATE_FIELDS:
            if field in state:
                current = getattr(agent, field)
                value = state[field]
                if isinstance(current, np.ndarray):
                    value = np.asarray(value, dtype=current.dtype)
                    if value.shape != current.shape:
                        raise ValueError(f'Checkpoint shape changed for {field}.')
                setattr(agent, field, value)


def load_checkpoint(path, expected_metadata):
    path = Path(path)
    if not path.exists():
        return None
    with path.open() as stream:
        saved = json.load(stream)
    if saved.get('schema_version') != 1 or saved.get('metadata') != expected_metadata:
        raise ValueError('Checkpoint settings or experiment inputs do not match this run.')
    return saved


def save_checkpoint(path, metadata, state):
    """Replace one complete file atomically, so interrupted writes cannot corrupt it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'schema_version': 1, 'metadata': metadata, **state}
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp',
                                         delete=False) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, default=json_value, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
