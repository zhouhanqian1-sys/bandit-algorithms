"""Execute the existing assignment notebook in ordered, restartable phases.

Example (inside the existing RL environment on an allocated compute node):
    python -u scripts/run_full_experiments.py --phase all --output-dir outputs/full_run

This runner never changes the notebook's T or N. It runs real notebook cells,
keeps completed per-agent comparisons, and stores figures and numerical results.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import traceback


PROJECT = Path(__file__).resolve().parents[1]
PHASES = ('parts1-6', '7a', '7b')


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def phase_indices(notebook, phase):
    """Use existing section headings, rather than hard-coding cell numbers."""
    section = None
    indices = []
    for index, cell in enumerate(notebook['cells']):
        source = ''.join(cell.get('source', []))
        if cell['cell_type'] == 'markdown':
            match = re.search(r'^## Part\s+(\d+[AB]?)\b', source, re.MULTILINE)
            if match:
                section = match.group(1)
        belongs = ((phase == 'parts1-6' and section in ('1', '2', '3', '4', '5', '6'))
                   or (phase == '7a' and section == '7A')
                   or (phase == '7b' and section == '7B'))
        if belongs and cell['cell_type'] == 'code':
            indices.append(index)
    if not indices:
        raise ValueError(f'No notebook code found for {phase}.')
    return indices


def source_fingerprint(notebook, indices):
    digest = hashlib.sha256()
    for index in indices:
        digest.update(''.join(notebook['cells'][index]['source']).encode())
    for folder in ('Model', 'algorithm', 'bandit'):
        for path in sorted((PROJECT / folder).glob('*.py')):
            digest.update(str(path.relative_to(PROJECT)).encode())
            digest.update(path.read_bytes())
    for relative in ('utils.py', 'scripts/checkpoint_support.py'):
        path = PROJECT / relative
        if path.exists():
            digest.update(relative.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


# Executed once in the notebook kernel. The original cells remain unchanged.
KERNEL_HELPERS = r'''
import hashlib as _assignment_hashlib
import json as _assignment_json
import os as _assignment_os
from pathlib import Path as _AssignmentPath
import time as _assignment_time
import numpy as _assignment_np
import matplotlib.pyplot as _assignment_plt
import utils as _assignment_utils

get_ipython().run_line_magic('matplotlib', 'inline')
_assignment_root = _AssignmentPath(ASSIGNMENT_PHASE_DIR)
_assignment_current_cell = -1
_assignment_figure_number = 0
_assignment_experiment_number = 0
_assignment_active_experiment = None
_assignment_agent_number = 0
_assignment_original_show = _assignment_plt.show
_assignment_original_run = _assignment_utils.run_experiment
_assignment_original_play = _assignment_utils.play_bandit
_assignment_reward_telemetry = {}
_telemetry_path = _assignment_root / 'telemetry.json'
if _telemetry_path.exists():
    _assignment_reward_telemetry = _assignment_json.loads(_telemetry_path.read_text())


def _assignment_write_json(path, value):
    path = _AssignmentPath(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(_assignment_json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def _assignment_save_npz(path, arrays):
    path = _AssignmentPath(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        _assignment_np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _assignment_json_value(value):
    if isinstance(value, _assignment_np.generic):
        return value.item()
    if isinstance(value, _assignment_np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _assignment_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_assignment_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(type(value).__name__)


def _assignment_show(*args, **kwargs):
    global _assignment_figure_number
    for number in _assignment_plt.get_fignums():
        figure = _assignment_plt.figure(number)
        _assignment_figure_number += 1
        # Long configuration labels need more room; numerical results are unaffected.
        width, height = figure.get_size_inches()
        if width < 10:
            figure.set_size_inches(10, max(height, 5))
        stem = f'cell_{_assignment_current_cell:03d}_figure_{_assignment_figure_number:03d}'
        figure.savefig(_assignment_root / 'figures' / (stem + '.png'), dpi=180, bbox_inches='tight')
        figure.savefig(_assignment_root / 'figures' / (stem + '.pdf'), bbox_inches='tight')
    return _assignment_original_show(*args, **kwargs)


_assignment_plt.show = _assignment_show


def _assignment_environment_signature(environment):
    signature = {'class': type(environment).__name__}
    for attribute in ('d', 'K', 'n_contexts', 'var'):
        if hasattr(environment, attribute):
            signature[attribute] = _assignment_json_value(getattr(environment, attribute))
    for attribute in ('theta', 'features'):
        if hasattr(environment, attribute):
            value = _assignment_np.asarray(getattr(environment, attribute))
            signature[attribute + '_sha256'] = _assignment_hashlib.sha256(value.tobytes()).hexdigest()
    dataset = getattr(environment, 'dataset', None)
    if dataset is not None:
        signature['dataset_length'] = len(dataset)
    return signature


def _assignment_play(environment, agent, N, T, seed=42):
    global _assignment_agent_number
    _assignment_agent_number += 1
    name = agent.name()
    key = f'{_assignment_active_experiment:03d}_agent_{_assignment_agent_number:02d}'
    path = _assignment_root / 'comparison_cache' / (key + '.npz')
    metadata_path = path.with_suffix('.json')
    expected = {'name': name, 'N': int(N), 'T': int(T), 'seed': seed,
                'environment': _assignment_environment_signature(environment)}
    if path.exists() and metadata_path.exists():
        previous = _assignment_json.loads(metadata_path.read_text())
        if previous['configuration'] != expected:
            raise ValueError(f'Cache configuration changed for {key}; use a new output directory.')
        with _assignment_np.load(path, allow_pickle=False) as saved:
            regrets = saved['regrets'].copy()
        if regrets.shape != (N, T):
            raise ValueError(f'Invalid cached regret shape for {key}.')
        print(f'RESUME experiment {_assignment_active_experiment}, {name}', flush=True)
        return name, regrets
    print(f'START experiment {_assignment_active_experiment}, {name}, N={N}, T={T}', flush=True)
    started = _assignment_time.monotonic()
    result_name, regrets = _assignment_original_play(environment, agent, N, T, seed=seed)
    if not _assignment_np.isfinite(regrets).all():
        raise FloatingPointError(f'Non-finite regret for {name}.')
    _assignment_save_npz(path, {'regrets': regrets})
    _assignment_write_json(metadata_path, {'configuration': expected,
                                          'seconds': _assignment_time.monotonic() - started})
    print(f'COMPLETE experiment {_assignment_active_experiment}, {name}', flush=True)
    return result_name, regrets


def _assignment_run(environment, agents, N, T, seed=42):
    global _assignment_experiment_number, _assignment_active_experiment, _assignment_agent_number
    _assignment_experiment_number += 1
    _assignment_active_experiment = _assignment_experiment_number
    _assignment_agent_number = 0
    return _assignment_original_run(environment, agents, N, T, seed=seed)


_assignment_utils.play_bandit = _assignment_play
_assignment_utils.run_experiment = _assignment_run


def _assignment_observe_reward(function, label):
    def observed(*args, **kwargs):
        statistics = _assignment_reward_telemetry.setdefault(
            label, {'successful_calls': 0, 'failed_calls': 0, 'seconds': 0.0})
        started = _assignment_time.monotonic()
        failed = False
        try:
            result = function(*args, **kwargs)
            statistics['successful_calls'] += 1
            return result
        except BaseException:
            failed = True
            statistics['failed_calls'] += 1
            raise
        finally:
            statistics['seconds'] += _assignment_time.monotonic() - started
            count = statistics['successful_calls']
            if failed or count % 100 == 0:
                _assignment_write_json(_telemetry_path, _assignment_reward_telemetry)
                print(f'LLM progress {label}: {count} successful calls, '
                      f"{statistics['seconds']:.1f}s inside reward generation", flush=True)
    observed._assignment_observed = True
    return observed


def _assignment_export(namespace):
    for reward_name in ('movie_reward', 'patient_reward'):
        function = namespace.get(reward_name)
        if callable(function) and not getattr(function, '_assignment_observed', False):
            namespace[reward_name] = _assignment_observe_reward(function, reward_name)
    _assignment_write_json(_telemetry_path, _assignment_reward_telemetry)
    arrays = {}
    manifest = {'cell_index': _assignment_current_cell, 'variables': {}, 'values': {}}
    summaries = {}
    for name, value in list(namespace.items()):
        if name.startswith('_') or name.startswith('ASSIGNMENT_'):
            continue
        if isinstance(value, _assignment_np.ndarray) and value.dtype.kind in 'biufc':
            arrays[name] = value
            manifest['variables'][name] = {'kind': 'array', 'key': name,
                                           'shape': list(value.shape), 'dtype': str(value.dtype)}
        elif isinstance(value, dict) and value and all(isinstance(item, _assignment_np.ndarray)
                                                       and item.dtype.kind in 'biufc'
                                                       for item in value.values()):
            items = {}
            for item_index, (algorithm, table) in enumerate(value.items()):
                key = f'{name}__{item_index}'
                arrays[key] = table
                items[algorithm] = key
                if table.ndim == 2:
                    final_values = table.sum(axis=1)
                    summaries.setdefault(name, {})[algorithm] = {
                        'N': int(table.shape[0]), 'T': int(table.shape[1]),
                        'final_mean': float(final_values.mean()),
                        'final_sample_std': float(final_values.std(ddof=1)) if len(final_values) > 1 else 0.0,
                        'final_percentile_10': float(_assignment_np.percentile(final_values, 10)),
                        'final_percentile_90': float(_assignment_np.percentile(final_values, 90)),
                    }
            manifest['variables'][name] = {'kind': 'dict', 'items': items}
        elif name == 'medical_features' and isinstance(value, list):
            items = []
            for item_index, table in enumerate(value):
                key = f'{name}__{item_index}'
                arrays[key] = table
                items.append(key)
            manifest['variables'][name] = {'kind': 'list', 'items': items}
        if name == 'medical_questions':
            manifest['values'][name] = [
                {'prompt_id': question['prompt_id'],
                 'answer_ids': [answer['answer_id'] for answer in question['answers']],
                 'answer_count': len(question['answers'])}
                for question in value]
        if (name.endswith('_summary') or name.startswith('best_')
            or name in ('movie_names', 'medical_names', 'movies', 'roles', 'question_ids',
                        'medical_profiles', 'lmcts_configurations',
                        'configuration_by_agent_name', 'movie_N', 'movie_T', 'medical_N', 'medical_T')):
            try:
                manifest['values'][name] = _assignment_json_value(value)
            except TypeError:
                pass
    # These experiments use rewards with shape (algorithm, run, round).
    for name, value in arrays.items():
        if name.startswith(('movie_rewards', 'medical_rewards')) and value.ndim == 3:
            names_key = 'movie_names' if name.startswith('movie_') else 'medical_names'
            algorithm_names = namespace.get(names_key, [])
            for index, algorithm in enumerate(algorithm_names):
                final_values = value[index].sum(axis=1)
                summaries.setdefault(name, {})[algorithm] = {
                    'N': int(value.shape[1]), 'T': int(value.shape[2]),
                    'final_mean': float(final_values.mean()),
                    'final_sample_std': float(final_values.std(ddof=1)) if len(final_values) > 1 else 0.0,
                    'final_percentile_10': float(_assignment_np.percentile(final_values, 10)),
                    'final_percentile_90': float(_assignment_np.percentile(final_values, 90)),
                }
    for name, value in list(arrays.items()):
        if name.startswith('movie_actions') and value.ndim == 3:
            counts = _assignment_np.zeros((value.shape[0], value.shape[1], len(namespace['movies'])), dtype=int)
            for algorithm in range(value.shape[0]):
                for run in range(value.shape[1]):
                    counts[algorithm, run] = _assignment_np.bincount(value[algorithm, run], minlength=counts.shape[2])
            key = name + '_counts'
            arrays[key] = counts
            manifest['variables'][key] = {'kind': 'array', 'key': key,
                                          'shape': list(counts.shape), 'dtype': str(counts.dtype)}
    _assignment_save_npz(_assignment_root / 'arrays.npz', arrays)
    _assignment_write_json(_assignment_root / 'manifest.json', manifest)
    _assignment_write_json(_assignment_root / 'summary.json', summaries)


def _assignment_validate_medical():
    assert len(medical_questions) == 5, 'The specified experiment requires five questions.'
    assert sum(len(question['answers']) for question in medical_questions) == 17, 'Expected 17 unique responses.'
    assert len(medical_features) == 5
    assert all(features.ndim == 2 and features.shape[1] == 17 for features in medical_features)
    print('MEDICAL_SETUP_VALIDATED: five questions, 17 responses, 17-dimensional features.', flush=True)
'''


def log_outputs(cell, log_path):
    with log_path.open('a') as stream:
        for output in cell.get('outputs', []):
            if output.get('output_type') == 'stream':
                continue  # Stream messages were already saved live by the client.
            elif output.get('output_type') == 'error':
                text = '\n'.join(output.get('traceback', [])) + '\n'
            else:
                text = output.get('data', {}).get('text/plain', '')
            if isinstance(text, list):
                text = ''.join(text)
            if text:
                stream.write(text)
                if not text.endswith('\n'):
                    stream.write('\n')
                print(text, end='' if text.endswith('\n') else '\n', flush=True)


def write_executed(notebook, path, original_length):
    import nbformat
    saved = deepcopy(notebook)
    saved['cells'] = saved['cells'][:original_length]
    temporary = path.with_name(path.name + '.tmp')
    nbformat.write(saved, temporary)
    temporary.replace(path)


def run_phase(arguments, original, phase):
    import nbformat
    from nbclient import NotebookClient

    phase_dir = arguments.output_dir.resolve() / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    for folder in ('figures', 'comparison_cache', 'checkpoints'):
        (phase_dir / folder).mkdir(exist_ok=True)
    indices = phase_indices(original, phase)
    fingerprint = source_fingerprint(original, indices)
    state_path = phase_dir / 'status.json'
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if previous.get('source_fingerprint') != fingerprint:
            raise ValueError(f'Source changed for {phase}; use a new output directory to avoid stale checkpoints.')
        if previous.get('state') == 'complete' and not arguments.rerun:
            print(f'SKIP completed phase {phase}', flush=True)
            return
    notebook = deepcopy(original)
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    original_length = len(notebook['cells'])
    notebook['cells'].append(nbformat.v4.new_code_cell(''))
    helper_index = original_length
    class StreamingNotebookClient(NotebookClient):
        def process_message(self, message, cell, cell_index):
            # Preserve progress if a long cell is interrupted before it returns.
            if message.get('msg_type') == 'stream':
                text = message['content'].get('text', '')
                with log_path.open('a') as stream:
                    stream.write(text)
                print(text, end='', flush=True)
            return super().process_message(message, cell, cell_index)

    client = StreamingNotebookClient(notebook, kernel_name=arguments.kernel, timeout=None,
                            allow_errors=False, iopub_timeout=120, record_timing=True)
    kernel_environment = dict(os.environ)
    kernel_environment['BIOSTAT825_CHECKPOINT_DIR'] = str(phase_dir / 'checkpoints')
    kernel_environment['PYTHONNOUSERSITE'] = '1'
    kernel_environment['PYTHONDONTWRITEBYTECODE'] = '1'
    state = {'phase': phase, 'state': 'running', 'started_at': now(),
             'source_fingerprint': fingerprint, 'completed_cells': [],
             'N_T_policy': 'Original notebook defaults; no overrides.'}
    atomic_json(state_path, state)
    log_path = phase_dir / 'execution.log'
    executed_path = phase_dir / 'Experiments.executed.ipynb'

    def auxiliary(source):
        notebook['cells'][helper_index] = nbformat.v4.new_code_cell(source)
        result = client.execute_cell(notebook['cells'][helper_index], helper_index)
        log_outputs(result, log_path)

    try:
        with client.setup_kernel(cwd=str(PROJECT), env=kernel_environment):
            auxiliary('ASSIGNMENT_PHASE_DIR = ' + repr(str(phase_dir)) + '\n' + KERNEL_HELPERS)
            for index in indices:
                cell = notebook['cells'][index]
                state.update(current_cell=index, current_cell_id=cell.get('id'))
                atomic_json(state_path, state)
                print(f'[{now()}] {phase}: executing cell {index} ({cell.get("id")})', flush=True)
                auxiliary(f'_assignment_current_cell = {index}')
                try:
                    client.execute_cell(cell, index)
                finally:
                    log_outputs(cell, log_path)
                    write_executed(notebook, executed_path, original_length)
                source = ''.join(cell['source'])
                if phase == '7b' and 'medical_features = encode_answers(medical_questions)' in source:
                    auxiliary('_assignment_validate_medical()')
                auxiliary('_assignment_export(globals())')
                state['completed_cells'].append(index)
                state['updated_at'] = now()
                atomic_json(state_path, state)
            state.update(state='complete', completed_at=now())
            atomic_json(state_path, state)
            print(f'COMPLETE phase {phase}', flush=True)
    except BaseException:
        state.update(state='failed', failed_at=now(), error=traceback.format_exc())
        atomic_json(state_path, state)
        write_executed(notebook, executed_path, original_length)
        raise


def combine_notebooks(original, output_dir):
    """Merge phase outputs by original cell ID, leaving section numbering intact."""
    import nbformat
    combined = deepcopy(original)
    cells_by_id = {cell.get('id'): cell for cell in combined['cells']}
    for phase in PHASES:
        path = output_dir / phase / 'Experiments.executed.ipynb'
        if not path.exists():
            continue
        executed = nbformat.read(path, as_version=4)
        for cell in executed['cells']:
            if cell.get('outputs') or cell.get('execution_count') is not None:
                target = cells_by_id.get(cell.get('id'))
                if target is not None:
                    target.update(outputs=cell.get('outputs', []), execution_count=cell.get('execution_count'))
                    target['metadata'].update(cell.get('metadata', {}))
    path = output_dir / 'Experiments.executed.ipynb'
    nbformat.write(combined, path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=(*PHASES, 'all'), default='all')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--notebook', type=Path, default=PROJECT / 'Experiments.ipynb')
    parser.add_argument('--kernel', default='RL')
    parser.add_argument('--rerun', action='store_true', help='Replay completed phases, retaining valid comparison caches.')
    arguments = parser.parse_args()
    import nbformat
    original = nbformat.read(arguments.notebook, as_version=4)
    nbformat.validate(original)
    arguments.output_dir = arguments.output_dir.resolve()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    phases = PHASES if arguments.phase == 'all' else (arguments.phase,)
    for phase in phases:
        run_phase(arguments, original, phase)
        combine_notebooks(original, arguments.output_dir)
    print(f'Executed notebook: {arguments.output_dir / "Experiments.executed.ipynb"}', flush=True)


if __name__ == '__main__':
    main()
