import ast
from functools import partial
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler


def source_functions():
    path = Path(__file__).resolve().parents[1] / "main_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    seed = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "seed_mcln_random_generators"
    )
    trainer = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BaseTrainTester"
    )
    loader = next(
        node for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_loaders"
    )
    namespace = {
        "random": random, "np": np, "torch": torch,
        "DataLoader": DataLoader,
        "DistributedSampler": partial(
            DistributedSampler, num_replicas=1, rank=0
        ),
        "is_counterfactual_parent_bounded_audit": lambda args: False,
        "_requires_joint_det_structured_collate": lambda args: False,
    }
    tree.body = [seed, loader]
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


class RandomDataset(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        return index, random.random(), np.random.rand(), torch.rand(())


def options(seed, evaluation):
    return SimpleNamespace(
        rng_seed=seed, eval=evaluation, batch_size=3, num_workers=2,
        dataloader_prefetch_factor=2, persistent_train_workers=False,
    )


def draw_initialization(namespace, seed):
    namespace["seed_mcln_random_generators"](seed)
    return (
        random.random(), np.random.rand(),
        torch.nn.Linear(3, 2).weight.detach().clone(),
    )


def test_seed_controls_python_numpy_and_model_initialization():
    namespace = source_functions()
    first = draw_initialization(namespace, 7)
    repeat = draw_initialization(namespace, 7)
    different = draw_initialization(namespace, 19)
    assert first[:2] == repeat[:2]
    assert torch.equal(first[2], repeat[2])
    assert first[0] != different[0]
    assert first[1] != different[1]
    assert not torch.equal(first[2], different[2])


def loader_rows(namespace, seed, evaluation):
    dataset = RandomDataset()
    instance = SimpleNamespace(get_datasets=lambda args: (dataset, dataset))
    train, test = namespace["get_loaders"](
        instance, options(seed, evaluation)
    )
    loader = test if evaluation else train
    return [
        tuple(component.tolist() for component in batch)
        for batch in loader
    ]


def test_seed_controls_training_order_and_worker_draws():
    namespace = source_functions()
    first = loader_rows(namespace, 7, False)
    repeat = loader_rows(namespace, 7, False)
    different = loader_rows(namespace, 19, False)
    assert first == repeat
    assert [row[0] for row in first] != [row[0] for row in different]
    assert first[0][1:] != different[0][1:]


def test_eval_order_stays_fixed_while_worker_seed_changes():
    namespace = source_functions()
    first = loader_rows(namespace, 7, True)
    repeat = loader_rows(namespace, 7, True)
    different = loader_rows(namespace, 19, True)
    assert first == repeat
    assert [row[0] for row in first] == [row[0] for row in different]
    assert first[0][1:] != different[0][1:]
