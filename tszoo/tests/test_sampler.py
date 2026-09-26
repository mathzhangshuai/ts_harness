import unittest
from unittest.mock import patch

import torch

from tszoo.data.sampler import BlockShuffleSampler
from tszoo.training.trainer import parse_args, training_loader


class SamplerTests(unittest.TestCase):
    def test_uneven_blocks_and_ranks_cover_once_without_padding(self):
        for size in (0, 1, 7, 19, 32):
            for block_size in (1, 4, 8, 64):
                for replicas in (1, 2, 3, 8):
                    with self.subTest(size=size, block=block_size, ranks=replicas):
                        samplers = [
                            BlockShuffleSampler(
                                range(size),
                                block_size=block_size,
                                num_replicas=replicas,
                                rank=rank,
                                seed=42,
                            )
                            for rank in range(replicas)
                        ]
                        for epoch in (0, 1):
                            for sampler in samplers:
                                sampler.set_epoch(epoch)
                            ranks = [list(sampler) for sampler in samplers]
                            self.assertEqual(
                                [len(indices) for indices in ranks],
                                [size // replicas] * replicas,
                            )
                            merged = [index for indices in ranks for index in indices]
                            self.assertEqual(len(merged), len(set(merged)))
                            self.assertTrue(set(merged).issubset(range(size)))
                            whole = BlockShuffleSampler(
                                range(size), block_size=block_size, seed=42
                            )
                            whole.set_epoch(epoch)
                            order = list(whole)
                            self.assertEqual(set(order), set(range(size)))
                            for rank, indices in enumerate(ranks):
                                self.assertEqual(
                                    indices, order[rank : len(merged) : replicas]
                                )

    def test_real_scale_allocates_only_block_indices_and_reads_current_batch(self):
        class LazyDataset:
            def __init__(self):
                self.reads = []

            def __len__(self):
                return 41_893_260

            def __getitem__(self, index):
                self.reads.append(index)
                return index

        args = parse_args(["--batch-size", "8", "--workers", "0"])
        dataset = LazyDataset()
        original = torch.randperm
        allocations = []

        def bounded_randperm(size, **kwargs):
            self.assertLessEqual(size, args.shuffle_block_size)
            allocations.append(size)
            return original(size, **kwargs)

        with patch("torch.randperm", side_effect=bounded_randperm):
            loader = training_loader(dataset, args)
            self.assertEqual(dataset.reads, [])
            iterator = iter(loader)
            first = next(iterator)
            second = next(iterator)
            self.assertEqual(dataset.reads, first + second)
            self.assertEqual(len(set(dataset.reads)), 16)
        self.assertEqual(allocations, [640, 65536])

    def test_epoch_is_reproducible_and_changes_order(self):
        sampler = BlockShuffleSampler(range(100), block_size=7, seed=9)
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(sampler))
        sampler.set_epoch(0)
        self.assertEqual(first, list(sampler))
