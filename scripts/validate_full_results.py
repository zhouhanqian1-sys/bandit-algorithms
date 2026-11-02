"""Independently validate saved full-size assignment outputs without rerunning them."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import traceback

import numpy as np


PHASES = ('parts1-6', '7a', '7b')
# Explicit original Monte Carlo sizes and comparison counts; all horizons are 1000.
REGRET_EXPERIMENTS = {
    'lin_experiment': (50, 4), 'epsilon_experiment': (50, 4),
    'part2_experiment': (50, 5), 'part3_experiment': (5, 4),
    'part4_linear_experiment': (5, 4), 'logistic_experiment': (5, 4),
    'quadratic_experiment': (5, 4), 'quadratic_neural_results': (10, 4),
    'shuttle_tuning_results': (10, 4), 'shuttle_results': (10, 4),
}


def read_json(path):
    return json.loads(Path(path).read_text())


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def expected_code_cells(notebook, phase):
    section = None
    indices = []
    for index, cell in enumerate(notebook['cells']):
        source = ''.join(cell.get('source', []))
        if cell['cell_type'] == 'markdown':
            match = re.search(r'^## Part\s+(\d+[AB]?)\b', source, re.MULTILINE)
            if match:
                section = match.group(1)
        belongs = ((phase == 'parts1-6' and section in ('1', '2', '3', '4', '5', '6'))
                   or (phase == '7a' and section == '7A') or (phase == '7b' and section == '7B'))
        if belongs and cell['cell_type'] == 'code':
            indices.append(index)
    return indices


def check_notebook(notebook, indices):
    for index in indices:
        cell = notebook['cells'][index]
        check(cell.get('execution_count') is not None, f'Unexecuted code cell {index}.')
    for cell in notebook['cells']:
        check(not any(output.get('output_type') == 'error' for output in cell.get('outputs', [])),
              'An executed notebook contains an error output.')


def check_statistics(table, summary, N, T, label):
    check(table.shape == (N, T), f'{label}: expected {(N, T)}, found {table.shape}.')
    check(summary['N'] == N and summary['T'] == T, f'{label}: summary dimensions differ.')
    totals = table.sum(axis=1)
    expected = {'final_mean': totals.mean(), 'final_sample_std': totals.std(ddof=1),
                'final_percentile_10': np.percentile(totals, 10),
                'final_percentile_90': np.percentile(totals, 90)}
    for field, value in expected.items():
        check(np.isclose(summary[field], value, rtol=1e-10, atol=1e-10), f'{label}: incorrect {field}.')


def check_manifest(manifest, arrays):
    for name, specification in manifest['variables'].items():
        kind = specification['kind']
        keys = ([specification['key']] if kind == 'array' else
                list(specification['items'].values()) if kind == 'dict' else specification['items'])
        for key in keys:
            check(key in arrays, f'{name}: missing NPZ key {key}.')
            check(np.isfinite(arrays[key]).all(), f'{name}: non-finite data.')
        if kind == 'array':
            value = arrays[specification['key']]
            check(list(value.shape) == specification['shape'], f'{name}: manifest shape differs.')
            check(str(value.dtype) == specification['dtype'], f'{name}: manifest dtype differs.')


def check_figures(directory, minimum):
    png = {path.stem: path for path in directory.glob('*.png')}
    pdf = {path.stem: path for path in directory.glob('*.pdf')}
    check(len(png) >= minimum and set(png) == set(pdf), 'Missing PNG/PDF figure pairs.')
    for stem in png:
        for path, signature in [(png[stem], b'\x89PNG\r\n\x1a\n'), (pdf[stem], b'%PDF-')]:
            with path.open('rb') as stream:
                check(stream.read(len(signature)) == signature, f'Invalid figure file {path.name}.')
    return len(png)


def check_regrets(manifest, arrays, summaries):
    values = manifest['values']
    tables = {}
    for variable, (N, policy_count) in REGRET_EXPERIMENTS.items():
        specification = manifest['variables'][variable]
        check(specification['kind'] == 'dict' and len(specification['items']) == policy_count,
              f'{variable}: wrong algorithm count.')
        tables[variable] = {name: arrays[key] for name, key in specification['items'].items()}
        check(set(summaries[variable]) == set(tables[variable]), f'{variable}: summary names differ.')
        for name, table in tables[variable].items():
            check((table >= -1e-12).all(), f'{variable}: negative pseudo-regret.')
            check_statistics(table, summaries[variable][name], N, 1000, f'{variable}/{name}')
    epsilon_means = {name: table.sum(axis=1).mean() for name, table in tables['epsilon_experiment'].items()}
    best_epsilon = min(epsilon_means, key=epsilon_means.get)
    check(values['best_epsilon_name'] == best_epsilon, 'Selected epsilon does not minimize measured regret.')
    choices = {re.search(r'eps=([^,]+)', name).group(1) for name in epsilon_means}
    check(choices == {'0.05', '1/d', '1/t', '0.2'}, 'Required epsilon choices differ.')
    token = re.search(r'eps=([^,]+)', best_epsilon).group(1)
    expected_choice = token if token in ('1/d', '1/t') else float(token)
    check(values['best_epsilon_choice'] == expected_choice, 'Selected epsilon scalar differs.')
    check(best_epsilon in tables['part2_experiment'], 'Selected epsilon missing from baseline comparison.')
    tuning_means = {name: table.sum(axis=1).mean() for name, table in tables['shuttle_tuning_results'].items()}
    best_lmcts = min(tuning_means, key=tuning_means.get)
    check(values['best_lmcts_name'] == best_lmcts, 'Selected LMCTS does not minimize tested tuning regret.')
    configuration = {'lr': float(re.search(r'lr=([^,]+)', best_lmcts).group(1)),
                     'beta_inv': float(re.search(r'beta_inv=([^,]+)', best_lmcts).group(1)),
                     'num_iter': int(re.search(r'steps=(\d+)', best_lmcts).group(1))}
    check(values['best_lmcts_configuration'] == configuration, 'Selected LMCTS scalar settings differ.')
    check(values['configuration_by_agent_name'][best_lmcts] == configuration, 'LMCTS configuration mapping differs.')
    return {'comparisons': len(tables), 'best_epsilon': expected_choice, 'best_lmcts': best_lmcts}


def check_rewards(phase, manifest, arrays, summaries):
    movie = phase == '7a'
    names = manifest['values']['movie_names' if movie else 'medical_names']
    policy_count = 4 if movie else 3
    check(len(names) == policy_count and len(set(names)) == policy_count, 'Incorrect LLM policy names.')
    if not movie:
        questions = manifest['values']['medical_questions']
        counts = np.array([question['answer_count'] for question in questions])
        check(len(questions) == 5 and counts.sum() == 17, 'Expected five questions and 17 responses.')
        features = manifest['variables']['medical_features']['items']
        check(len(features) == 5, 'Missing medical feature matrices.')
        for count, key in zip(counts, features):
            check(arrays[key].shape == (count, 17), 'Incorrect medical feature construction shape.')
        check(np.array_equal(arrays['medical_question_indices_a'], arrays['medical_question_indices_b']),
              'Medical A/B question schedules differ.')
    for suffix in (('', '_b') if movie else ('_a', '_b')):
        reward_name = ('movie_rewards' if movie else 'medical_rewards') + suffix
        action_name = ('movie_actions' if movie else 'medical_actions') + suffix
        rewards, actions = arrays[reward_name], arrays[action_name]
        check(rewards.shape == actions.shape == (policy_count, 3, 1000), f'{reward_name}: original N/T changed.')
        check((rewards >= 0).all() and (rewards <= 1).all(), f'{reward_name}: reward outside [0, 1].')
        check(actions.dtype.kind in 'iu' and (actions >= 0).all(), f'{action_name}: invalid action indices.')
        if movie:
            check((actions < 8).all(), f'{action_name}: movie index outside 0..7.')
            saved_counts = arrays[action_name + '_counts']
            check(saved_counts.shape == (4, 3, 8) and (saved_counts.sum(axis=2) == 1000).all(), 'Movie counts differ.')
            for algorithm in range(4):
                for run in range(3):
                    check(np.array_equal(saved_counts[algorithm, run], np.bincount(actions[algorithm, run], minlength=8)),
                          'Movie counts do not match actual selected movies.')
        else:
            questions = arrays['medical_question_indices' + suffix]
            check(questions.shape == (3, 1000) and questions.dtype.kind in 'iu', 'Invalid question schedule.')
            check((questions >= 0).all() and (questions < 5).all(), 'Question index outside 0..4.')
            check((actions < counts[questions][None, :, :]).all(), 'A selected response is unavailable for its question.')
            check(np.allclose(rewards * 10, np.rint(rewards * 10), rtol=0, atol=1e-10), 'Medical reward is not score / 10.')
        check(set(summaries[reward_name]) == set(names), f'{reward_name}: summary policy names differ.')
        for algorithm, name in enumerate(names):
            check_statistics(rewards[algorithm], summaries[reward_name][name], 3, 1000, f'{reward_name}/{name}')
            for other in range(algorithm):
                same_choice = actions[algorithm] == actions[other]
                check(np.array_equal(rewards[algorithm][same_choice], rewards[other][same_choice]),
                      f'{reward_name}: identical same-round choices did not share rewards.')
    return {'profiles_or_roles': 2, 'policies': policy_count, 'N': 3, 'T': 1000}


def validate_phase(output_dir, phase):
    directory = output_dir / phase
    status = read_json(directory / 'status.json')
    check(status['state'] == 'complete', f'{phase} has not finished; current state is {status["state"]}.')
    notebook = read_json(directory / 'Experiments.executed.ipynb')
    indices = expected_code_cells(notebook, phase)
    check(indices, f'{phase}: no expected code cells.')
    check_notebook(notebook, indices)
    check(set(indices) == set(status['completed_cells']), f'{phase}: completion ledger differs from notebook.')
    manifest, summary = read_json(directory / 'manifest.json'), read_json(directory / 'summary.json')
    with np.load(directory / 'arrays.npz', allow_pickle=False) as arrays:
        check_manifest(manifest, arrays)
        details = (check_regrets(manifest, arrays, summary) if phase == 'parts1-6'
                   else check_rewards(phase, manifest, arrays, summary))
    details.update(passed=True, executed_code_cells=len(indices),
                   figure_pairs=check_figures(directory / 'figures', {'parts1-6': 10, '7a': 4, '7b': 2}[phase]))
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=PHASES, help='Check one completed phase; default requires all phases.')
    arguments = parser.parse_args()
    report = {'passed': False, 'all_phases_validated': False,
              'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'phases': {}}
    phases = (arguments.phase,) if arguments.phase else PHASES
    try:
        for phase in phases:
            report['phases'][phase] = validate_phase(arguments.output_dir, phase)
            print(f'PASS {phase}', flush=True)
        if arguments.phase is None:
            notebook = read_json(arguments.output_dir / 'Experiments.executed.ipynb')
            indices = [i for i, cell in enumerate(notebook['cells']) if cell['cell_type'] == 'code']
            check(len(indices) == 43, 'The combined notebook must contain all 43 original code cells.')
            check_notebook(notebook, indices)
            report['combined_executed_code_cells'] = len(indices)
        report['passed'] = True
        report['all_phases_validated'] = arguments.phase is None
    except Exception:
        report['error'] = traceback.format_exc()
    filename = f'validation-{arguments.phase}.json' if arguments.phase else 'validation-report.json'
    (arguments.output_dir / filename).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
