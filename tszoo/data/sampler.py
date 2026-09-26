"""Epoch shuffling without allocating an index for every training window."""

import torch
from torch.utils.data import DistributedSampler


class BlockShuffleSampler(DistributedSampler):
    """Shuffle blocks and their contents, then shard stream positions by rank.

    This is not a uniform global permutation. Index storage is bounded by one
    block plus the block order. Lightning owns set_epoch and DDP process launch.
    """

    def __init__(self, dataset, *, block_size=65536, num_replicas=1, rank=0, seed=0):
        if type(block_size) is not int or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )
        self.block_size = block_size

    def __iter__(self):
        size = len(self.dataset)
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        blocks = torch.randperm(
            (size + self.block_size - 1) // self.block_size, generator=generator
        )
        consumed = 0
        for block in blocks:
            if consumed >= self.total_size:
                break
            start = int(block) * self.block_size
            length = min(self.block_size, size - start)
            order = torch.randperm(length, generator=generator)
            stop = min(length, self.total_size - consumed)
            first = (self.rank - consumed) % self.num_replicas
            for position in range(first, stop, self.num_replicas):
                yield start + int(order[position])
            consumed += length
            del order
