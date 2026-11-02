"""Lightweight, local notebook checks; never execute experiment or download cells.

Run with any existing Python interpreter. NumPy checks and CPU PyTorch checks
are skipped when their dependencies are unavailable; no packages are installed.
LLM runner checks use test-only reward functions in an isolated namespace.
"""
import argparse
import ast
from collections import Counter
from copy import deepcopy
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
from types import SimpleNamespace


PROJECT = Path(__file__).resolve().parents[1]
NUMPY_CLASSES = {
    'LinEpsilonGreedy', 'LinUCB', 'LinTS', 'EpsilonLinTS',
    'EpsilonGreedy', 'UCB', 'Random', 'MedicalLinearBandit',
    'MedicalLinUCB', 'MedicalLinTS', 'MedicalRandom',
}


def read_notebook(path):
    notebook = json.loads(path.read_text())
    assert notebook['nbformat'] == 4
    assert isinstance(notebook['cells'], list)
    return notebook


def definition_nodes(notebook, ignore_syntax_errors=False):
    """Extract definitions only; top-level experiment and loading calls stay inert."""
    nodes = []
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        try:
            tree = ast.parse(''.join(cell['source']))
        except SyntaxError:
            if ignore_syntax_errors:
                continue
            raise
        nodes.extend(node for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.ClassDef)))
    return nodes


def literal_values(notebook):
    values = {}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        try:
            tree = ast.parse(''.join(cell['source']))
        except SyntaxError:
            # The original assignment contains intentionally unfinished code cells.
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        values[target.id] = value
    return values


def check_structure(notebook):
    headings = []
    code_count = 0
    for index, cell in enumerate(notebook['cells']):
        assert cell['cell_type'] in ('code', 'markdown', 'raw')
        source = ''.join(cell['source'])
        assert not re.search(r'[\u4e00-\u9fff]', source), f'Non-English text in cell {index}'
        if cell['cell_type'] == 'markdown':
            headings.extend(re.findall(r'^## Part\s+(\d+[AB]?)\b', source, re.MULTILINE))
        if cell['cell_type'] == 'code':
            ast.parse(source, filename=f'notebook-cell-{index}')
            assert not cell.get('outputs'), 'Code implementation must not include fabricated outputs.'
            assert cell.get('execution_count') is None
            assert 'TODO' not in source and 'YOUR CODE HERE' not in source
            code_count += 1
    assert headings == ['1', '2', '3', '4', '5', '6', '7A', '7B'], headings
    classes = [node.name for node in definition_nodes(notebook) if isinstance(node, ast.ClassDef)]
    assert not [name for name, count in Counter(classes).items() if count > 1], classes

    for folder in ['Model', 'algorithm', 'bandit', 'tests']:
        for path in (PROJECT / folder).glob('*.py'):
            source = path.read_text()
            ast.parse(source, filename=str(path))
            assert not re.search(r'[\u4e00-\u9fff]', source)
    ast.parse((PROJECT / 'utils.py').read_text())

    # Preserve literal assignment data, rather than calling any model or dataset loader.
    try:
        original = json.loads(subprocess.check_output(
            ['git', 'show', '3b3ab51:Experiments.ipynb'], cwd=PROJECT, text=True))
    except (OSError, subprocess.CalledProcessError):
        original = None
    current_values = literal_values(notebook)
    if original is not None:
        original_values = literal_values(original)
        for name in ['movies', 'roles', 'medical_profiles', 'question_ids',
                     'movie_model_id', 'movie_revision', 'patient_model_id',
                     'patient_revision', 'encoder_model_id', 'encoder_revision']:
            assert current_values[name] == original_values[name], name
        original_functions = {node.name: node for node in definition_nodes(original, ignore_syntax_errors=True)
                              if isinstance(node, ast.FunctionDef)}
        current_functions = {node.name: node for node in definition_nodes(notebook)
                             if isinstance(node, ast.FunctionDef)}
        assert ast.dump(current_functions['encode_answers']) == ast.dump(original_functions['encode_answers'])
    assert current_values['movie_N'] == 3 and current_values['movie_T'] == 1000
    assert current_values['medical_N'] == 3 and current_values['medical_T'] == 1000
    return f'{code_count} code cells parse; eight headings, English content, model/profile literals preserved'


def load_definitions(notebook, np, with_torch=False):
    namespace = {'np': np, 'Agent': object, 'deepcopy': deepcopy,
                 'random': random, 'math': math, 'log': math.log,
                 'pinv': np.linalg.pinv}
    if with_torch:
        import torch
        import torch.nn as nn
        from torch.optim import SGD
        from Model.MLP import MLP
        from Model.LinearModel import LinearModel
        from algorithm.optim import LangevinMC
        from utils import Collector, run_experiment, plot_result
        namespace.update(torch=torch, nn=nn, SGD=SGD, MLP=MLP, LinearModel=LinearModel,
                         LangevinMC=LangevinMC, Collector=Collector,
                         run_experiment=run_experiment, plot_result=plot_result)
    else:
        # Only mocked reward callbacks use this stub; no Torch algorithm is tested with it.
        namespace['torch'] = SimpleNamespace(manual_seed=lambda seed: None)
    nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    for node in definition_nodes(notebook):
        if isinstance(node, ast.FunctionDef) or with_torch or node.name in NUMPY_CLASSES:
            nodes.append(node)
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, '<notebook-definitions>', 'exec'), namespace)
    return namespace


def check_linear(namespace, np):
    arms = np.eye(3)
    observations = [np.array([1.0, 0.5, 0.0]), np.array([0.0, 1.0, 1.0])]
    rewards = [0.25, 0.75]
    agents = [namespace['LinEpsilonGreedy'](3, lambda_reg=2.0, eps=0.0),
              namespace['LinUCB'](3, delta=0.2, lambda_reg=2.0),
              namespace['LinTS'](3, delta=0.2, lambda_prior=2.0),
              namespace['EpsilonLinTS'](3, delta=0.2, lambda_prior=2.0, eps='1/t')]
    expected_cov = 2 * np.eye(3)
    expected_target = np.zeros(3)
    for x, reward in zip(observations, rewards):
        expected_cov += np.outer(x, x)
        expected_target += x * reward
        for agent in agents:
            index, feature = agent.get_action(arms)
            assert 0 <= index < len(arms) and np.array_equal(feature, arms[index])
            agent.receive_reward(x, x, reward)
            assert np.allclose(agent.cov, expected_cov)
            assert np.allclose(agent.invcov, np.linalg.inv(expected_cov))
            assert np.allclose(agent.hat_theta, np.linalg.solve(expected_cov, expected_target))
    for agent in agents:
        agent.reset()
        assert agent.t == 0 and not agent.b_t.any() and not agent.hat_theta.any()
        assert np.allclose(agent.cov, 2 * np.eye(3))
    epsilon_agent = namespace['EpsilonLinTS'](3, 0.2, 1.0, eps='1/t')
    assert epsilon_agent.exploration_probability() == 1.0
    epsilon_agent.receive_reward(arms[0], arms[0], 1.0)
    assert epsilon_agent.exploration_probability() == 0.5
    assert namespace['EpsilonLinTS'](3, 0.2, 1.0, eps='1/d').exploration_probability() == 1 / 3
    assert len({namespace['EpsilonLinTS'](3, 0.2, 1.0, eps=eps).name()
                for eps in [0.05, '1/d', '1/t', 0.2]}) == 4
    greedy = namespace['LinEpsilonGreedy'](3, eps=0)
    greedy.hat_theta = np.array([0.1, 0.7, 0.2])
    assert greedy.get_action(arms)[0] == 1
    ucb = namespace['LinUCB'](3, 0.2, 1.0)
    ucb.hat_theta = np.array([0.7, 0.0, 0.0])
    ucb.invcov = np.diag([0.01, 100.0, 0.01])
    assert ucb.get_action(arms)[0] == 1, 'UCB must account for uncertainty.'
    ts = namespace['LinTS'](3, 0.2, 1.0)
    np.random.seed(11)
    chosen = {ts.get_action(np.array([[1., 0., 0.], [-1., 0., 0.]]))[0] for _ in range(32)}
    assert chosen == {0, 1}, 'Posterior samples must affect Thompson actions.'
    return 'ridge/inverse updates, action rules, posterior sampling, one-based epsilon, resets and names'


def check_llm_runners(namespace, np):
    EpsilonGreedy = namespace['EpsilonGreedy']
    UCB = namespace['UCB']
    MedicalLinearBandit = namespace['MedicalLinearBandit']
    MedicalLinUCB = namespace['MedicalLinUCB']
    MedicalLinTS = namespace['MedicalLinTS']

    agent = EpsilonGreedy(3, eps=0, seed=13)
    agent.receive_reward(1, 0.25)
    agent.receive_reward(1, 0.75)
    assert agent.counts.tolist() == [0, 2, 0]
    assert np.allclose(agent.mean_rewards, [0, 0.5, 0])
    assert agent.get_action() == 1
    agent.reset()
    assert agent.t == 0 and not agent.counts.any() and not agent.mean_rewards.any()
    ucb = UCB(3, seed=20)
    visited = []
    for _ in range(3):
        arm = ucb.get_action()
        visited.append(arm)
        ucb.receive_reward(arm, 0.5)
    assert len(set(visited)) == 3
    assert UCB(3, 2).name() != UCB(3, 1).name()

    base = MedicalLinearBandit(3, lambda_reg=2.0)
    xs = [np.array([1., 0.5, 0.]), np.array([0., 1., 1.])]
    ys = [0.25, 0.75]
    for x, y in zip(xs, ys):
        base.receive_reward(x, y)
    expected_cov = 2 * np.eye(3) + sum(np.outer(x, x) for x in xs)
    expected_target = sum(x*y for x, y in zip(xs, ys))
    assert np.allclose(base.cov, expected_cov)
    assert np.allclose(base.invcov, np.linalg.inv(expected_cov))
    assert np.allclose(base.hat_theta, np.linalg.solve(expected_cov, expected_target))
    base.reset()
    assert base.t == 0 and np.allclose(base.cov, 2*np.eye(3)) and not base.b_t.any()
    assert MedicalLinUCB(3, alpha=0.5).name() != MedicalLinUCB(3, alpha=1).name()
    assert MedicalLinTS(3, nu=0.5).name() != MedicalLinTS(3, nu=1).name()

    movie_calls = []
    last_torch_seed = [None]
    namespace['torch'].manual_seed = lambda seed: last_torch_seed.__setitem__(0, seed)
    namespace['movies'] = [('a',''), ('b',''), ('c','')]
    namespace['roles'] = {'A':'space', 'B':'musicals'}
    def mock_movie_reward(index, role):
        movie_calls.append((index, role, last_torch_seed[0]))
        return float(np.random.uniform())
    namespace['movie_reward'] = mock_movie_reward
    run_movies = namespace['run_movie_bandits']
    initial_global_state = np.random.get_state()
    movie_result = run_movies('A', N=2, T=12, seed=41)
    assert movie_result[0].shape == (4, 2, 12)
    assert movie_result[1].shape == (4, 2, 12)
    assert np.array_equal(np.random.get_state()[1], initial_global_state[1])
    expected_calls = 0
    for run in range(2):
        for time in range(12):
            expected_calls += len(set(movie_result[1][:, run, time]))
            for first in range(4):
                for second in range(4):
                    if movie_result[1][first,run,time] == movie_result[1][second,run,time]:
                        assert movie_result[0][first,run,time] == movie_result[0][second,run,time]
    assert len(movie_calls) == expected_calls
    assert len({seed for _,_,seed in movie_calls}) == len(movie_calls)
    movie_calls.clear()
    repeated = run_movies('A', N=2, T=12, seed=41)
    assert all(np.array_equal(left, right) for left, right in zip(movie_result, repeated))
    assert not np.array_equal(movie_result[1][:,0], movie_result[1][:,1])

    medical_calls = []
    namespace['medical_questions'] = [
        {'prompt_id':'q0', 'answers':[{'text':'a'},{'text':'b'}]},
        {'prompt_id':'q1', 'answers':[{'text':'c'},{'text':'d'},{'text':'e'}]},
    ]
    namespace['medical_features'] = [np.eye(3)[:2], np.eye(3)]
    namespace['medical_profiles'] = {'A':'brief','B':'detail'}
    def mock_patient_reward(question, profile, arm, seed):
        medical_calls.append((question['prompt_id'], profile, arm, seed))
        return float(np.random.default_rng(seed).uniform())
    namespace['patient_reward'] = mock_patient_reward
    run_medical = namespace['run_medical_bandits']
    medical_result = run_medical('A', N=2, T=12, seed=43)
    assert medical_result[0].shape == (3,2,12)
    assert medical_result[1].shape == (3,2,12)
    assert medical_result[2].shape == (2,12)
    expected_calls = 0
    for run in range(2):
        for time in range(12):
            question_index = medical_result[2][run,time]
            assert (medical_result[1][:,run,time] < len(namespace['medical_features'][question_index])).all()
            expected_calls += len(set(medical_result[1][:,run,time]))
            for first in range(3):
                for second in range(3):
                    if medical_result[1][first,run,time] == medical_result[1][second,run,time]:
                        assert medical_result[0][first,run,time] == medical_result[0][second,run,time]
    assert len(medical_calls) == expected_calls
    assert len({seed for _,_,_,seed in medical_calls}) == len(medical_calls)
    medical_calls.clear()
    repeated_medical = run_medical('A', N=2, T=12, seed=43)
    assert all(np.array_equal(left,right) for left,right in zip(medical_result,repeated_medical))
    profile_b = run_medical('B', N=2, T=12, seed=43)
    assert np.array_equal(medical_result[2], profile_b[2])
    assert not np.array_equal(medical_result[1][:,0], medical_result[1][:,1])

    class ConstantAgent:
        def reset(self, seed=None):
            self.observations = []
        def get_action(self, arms=None):
            return 0
        def receive_reward(self, arm, reward):
            self.observations.append(reward)
    movie_constant_agents = [ConstantAgent() for _ in range(4)]
    namespace['make_movie_agents'] = lambda num_arms: movie_constant_agents
    movie_calls.clear()
    shared_movies = run_movies('A', N=2, T=5, seed=52)
    assert len(movie_calls) == 10
    assert np.all(shared_movies[0] == shared_movies[0][0])
    assert all(len(agent.observations)==5 for agent in movie_constant_agents)
    assert len(np.unique(shared_movies[0])) > 1
    medical_constant_agents = [ConstantAgent() for _ in range(3)]
    namespace['make_medical_agents'] = lambda d: medical_constant_agents
    namespace['medical_questions'] = namespace['medical_questions'][:1]
    namespace['medical_features'] = namespace['medical_features'][:1]
    medical_calls.clear()
    shared_medical = run_medical('A', N=2, T=5, seed=53)
    assert len(medical_calls) == 10
    assert np.all(shared_medical[0] == shared_medical[0][0])
    assert all(len(agent.observations)==5 for agent in medical_constant_agents)
    assert len(np.unique(shared_medical[0])) > 1

    return 'sample means/UCB, medical ridge updates, runner shapes, seeded independence and fresh per-round shared rewards'


def check_torch(notebook, np):
    """Use CPU tensors and tiny, generated data; never construct a downloading loader."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.setdefault('MPLBACKEND', 'Agg')
    import torch
    from torch import nn
    from torch.optim import SGD
    from Model.MLP import MLP
    from Model.LinearModel import LinearModel
    from algorithm.optim import LangevinMC
    from utils import Collector, run_experiment
    from bandit.linear_bandit import LinearBandit
    from bandit.uci_bandit import UCIBandit
    namespace = load_definitions(notebook, np, with_torch=True)
    torch.set_num_threads(1)
    torch.manual_seed(17)

    collector = Collector()
    vector = np.array([1., 2., 3.])
    collector.collect_data(0, vector, 0.5)
    vector[0] = 99
    assert collector.context[0][0].item() == 1
    assert len(collector.fetch_batch(1)[0]) == 1
    collector.collect_data(0, torch.ones(3, requires_grad=True), 1.0)
    assert not collector.context[-1].requires_grad
    assert len(collector.fetch_batch(1)[0]) == 1
    np.random.seed(37)
    sampled_rewards = {collector.fetch_batch(1)[1][0] for _ in range(16)}
    assert sampled_rewards == {0.5, 1.0}, 'Both legal minibatch offsets must be reachable.'
    collector.clear()
    assert len(collector) == 0

    # Regression: a parameter may receive its first gradient after the first step.
    first = nn.Parameter(torch.ones(1, dtype=torch.float64))
    second = nn.Parameter(torch.ones(1, dtype=torch.float64))
    sampler = LangevinMC([first, second], lr=0.1, beta_inv=0, weight_decay=0, device='cpu')
    first.grad = torch.ones_like(first)
    sampler.step()
    second.grad = torch.ones_like(second)
    sampler.step()
    assert torch.allclose(first, torch.tensor([0.8], dtype=torch.float64))
    assert torch.allclose(second, torch.tensor([0.9], dtype=torch.float64))
    assert first.grad.item() == 1.0
    noisy_parameter = nn.Parameter(torch.zeros(4, dtype=torch.float64))
    noisy_parameter.grad = torch.zeros_like(noisy_parameter)
    noisy_sampler = LangevinMC([noisy_parameter], lr=0.05, beta_inv=0.2,
                              sigma=0.4, weight_decay=0, device='cpu')
    torch.manual_seed(19)
    expected_noise = 0.4 * math.sqrt(2 * 0.05 * 0.2) * torch.randn_like(noisy_parameter)
    torch.manual_seed(19)
    noisy_sampler.step()
    assert torch.allclose(noisy_parameter, expected_noise)

    arms = np.eye(3, dtype=np.float64)
    template = MLP(1, 3, layers=[4]).double()
    template_optimizer = SGD(template.parameters(), lr=0.01, momentum=0.9)
    initial_template = [parameter.detach().clone() for parameter in template.parameters()]
    common = dict(d=3, model=template, optimizer=template_optimizer,
                  criterion=nn.MSELoss(), collector=Collector(), batch_size=None, device='cpu')
    agents = [namespace['NeuralEpsGreedy'](**common, eps=0.1, num_iter=1),
              namespace['NeuralTS'](**common, nu=0.2, reg=1.0, image=False, num_iter=1),
              namespace['NeuralUCB'](**common, nu=0.2, reg=1.0, image=False)]
    linear_model = LinearModel(3, 1).double()
    langevin = LangevinMC(linear_model.parameters(), lr=0.01, beta_inv=0.01, device='cpu')
    agents.append(namespace['LMCTS'](3, linear_model, langevin, nn.MSELoss(),
                                    Collector(), None, 'cpu', 1))
    neural_sampler = LangevinMC(template.parameters(), lr=0.01, beta_inv=0.01, device='cpu')
    agents.append(namespace['NeuralLMCTS'](3, template, neural_sampler, nn.MSELoss(),
                                          Collector(), None, 'cpu', num_iter=1))
    parameter_addresses = []
    for agent in agents:
        model_parameters = list(agent.model.parameters())
        optimizer_parameters = [parameter for group in agent.optimizer.param_groups
                                for parameter in group['params']]
        assert {id(p) for p in model_parameters} == {id(p) for p in optimizer_parameters}
        parameter_addresses.extend(parameter.data_ptr() for parameter in model_parameters)
        index, arm = agent.get_action(arms)
        assert np.array_equal(arm, arms[index])
        selected_gradient = getattr(agent, 'selected_gradient', None)
        previous_precision = agent.precision.clone() if hasattr(agent, 'precision') else None
        agent.receive_reward(arm, arm, 0.5)
        assert agent.t == 1 and len(agent.collector) == 1
        assert math.isfinite(agent.last_loss)
        if previous_precision is not None:
            assert torch.allclose(agent.precision, previous_precision + selected_gradient.square())
            assert torch.all(agent.precision > 0)
        assert all(parameter.dtype == torch.float64 for parameter in model_parameters)
        agent.reset()
        assert agent.t == 0 and len(agent.collector) == 0 and not agent.optimizer.state
        if hasattr(agent, 'precision'):
            assert torch.all(agent.precision == agent.reg)
        if isinstance(agent.optimizer, LangevinMC):
            assert agent.optimizer.curr_step == 0
    assert len(parameter_addresses) == len(set(parameter_addresses))
    assert all(torch.equal(old, new) for old, new in zip(initial_template, template.parameters()))

    # Exercise the actual shuttle-sized factory and runner without loading shuttle data.
    factory_agents = namespace['make_neural_agents'](63, 'cpu')
    factory_environment = LinearBandit(np.ones(63) / np.sqrt(63), 7, 2)
    factory_results = run_experiment(factory_environment, factory_agents, N=2, T=3, seed=28)
    assert len(factory_results) == 4
    for agent in factory_agents:
        regrets = factory_results[agent.name()]
        assert regrets.shape == (2, 3)
        assert np.isfinite(regrets).all() and (regrets >= 0).all()
        assert agent.t == 3 and len(agent.collector) == 3

    class LoggingAgent:
        def __init__(self, label, extra_draws):
            self.label = label
            self.extra_draws = extra_draws
            self.contexts = []
            self.rewards = []
        def reset(self):
            self.contexts.append([])
            self.rewards.append([])
        def get_action(self, features):
            np.random.random(self.extra_draws)
            torch.rand(self.extra_draws)
            self.contexts[-1].append(features.copy())
            return 0, features[0]
        def receive_reward(self, arm, context, reward):
            self.rewards[-1].append(reward)
        def name(self):
            return self.label

    env = LinearBandit(np.ones(3) / np.sqrt(3), 2, 5)
    left, right = LoggingAgent('few draws', 1), LoggingAgent('many draws', 7)
    numpy_state, python_state, torch_state = np.random.get_state(), random.getstate(), torch.get_rng_state()
    result = run_experiment(env, [left, right], N=2, T=4, seed=29)
    assert np.array_equal(result[left.name()], result[right.name()])
    assert np.array_equal(np.array(left.contexts), np.array(right.contexts))
    assert np.array_equal(np.array(left.rewards), np.array(right.rewards))
    assert len(left.contexts) == 2
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])
    assert numpy_state[2:] == np.random.get_state()[2:]
    assert python_state == random.getstate() and torch.equal(torch_state, torch.get_rng_state())
    try:
        run_experiment(env, [left, left], N=1, T=1)
        raise AssertionError('Duplicate result names must be rejected.')
    except ValueError:
        pass

    # Exercise UCI resetting without calling its network-loading constructor.
    uci = UCIBandit.__new__(UCIBandit)
    feature_rows = torch.zeros(4, 7, 63)
    feature_rows[:, :, 0] = torch.arange(1., 5.).unsqueeze(1)
    uci.loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(feature_rows, torch.arange(4)), batch_size=1, shuffle=True)
    uci.device = 'cpu'
    uci.K = 7
    uci.reset()
    assert uci.current_label is None and uci.current_action_set is None
    assert uci.get_action_set().shape == (7, 63)
    assert uci.get_reward(uci.current_label) == 1
    left, right = LoggingAgent('UCI few draws', 1), LoggingAgent('UCI many draws', 7)
    result = run_experiment(uci, [left, right], N=2, T=4, seed=31)
    assert np.array_equal(result[left.name()], result[right.name()])
    assert np.array_equal(np.array(left.contexts), np.array(right.contexts))
    return 'CPU model/optimizer ownership, dtype conversion, gradients/precision, resets, 63-input factory integration, collector bounds, fair RNG streams and UCI reset'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', type=Path, default=PROJECT / 'Experiments.ipynb')
    arguments = parser.parse_args()
    sys.path.insert(0, str(PROJECT))
    notebook = read_notebook(arguments.notebook)
    print('PASS structure:', check_structure(notebook))
    try:
        import numpy as np
    except ImportError:
        print('SKIP NumPy checks: NumPy is not installed in this interpreter.')
        print('SKIP CPU PyTorch checks: required local dependencies are unavailable.')
        return
    # Test-only rewards must not read or write production experiment checkpoints.
    os.environ.pop('BIOSTAT825_CHECKPOINT_DIR', None)
    namespace = load_definitions(notebook, np)
    print('PASS linear algorithms:', check_linear(namespace, np))
    print('PASS LLM runner logic:', check_llm_runners(namespace, np))
    required = ['torch', 'scipy', 'matplotlib', 'sklearn', 'seaborn']
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        print('SKIP CPU PyTorch checks: unavailable local dependencies:', ', '.join(missing))
    else:
        print('PASS CPU PyTorch/supporting code:', check_torch(notebook, np))
    print('Validation used definitions and tiny synthetic examples only; no full experiment, model, or dataset was loaded.')


if __name__ == '__main__':
    main()
