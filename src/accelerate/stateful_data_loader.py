from torchdata.stateful_dataloader import StatefulDataLoader

import math
from contextlib import suppress
from typing import Callable, List, Optional, Union

import torch
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, RandomSampler

from .data_loader import DataLoaderStateMixin, DataLoaderDispatcher, DataLoaderShard, SkipDataLoader, get_sampler
from .logging import get_logger
from .state import AcceleratorState, DistributedType, GradientState, is_torch_xla_available
from .utils import (
    RNGType,
    broadcast,
    broadcast_object_list,
    concatenate,
    find_batch_size,
    get_data_structure,
    initialize_tensors,
    is_torch_version,
    send_to_device,
    slice_tensors,
    synchronize_rng_states,
)

class StatefulDataLoaderShard(StatefulDataLoader, DataLoaderStateMixin):
    """
    Subclass of a torchdata `StatefulDataLoader` that will deal with device placement and current distributed setup.

    Args:
        dataset (`torch.utils.data.dataset.Dataset`):
            The dataset to use to build this datalaoder.
        device (`torch.device`, *optional*):
            If passed, the device to put all batches on.
        rng_types (list of `str` or [`~utils.RNGType`]):
            The list of random number generators to synchronize at the beginning of each iteration. Should be one or
            several of:

            - `"torch"`: the base torch random number generator
            - `"cuda"`: the CUDA random number generator (GPU only)
            - `"xla"`: the XLA random number generator (TPU only)
            - `"generator"`: an optional `torch.Generator`
        synchronized_generator (`torch.Generator`, *optional*):
            A random number generator to keep synchronized across processes.
        skip_batches (`int`, *optional*, defaults to 0):
            The number of batches to skip at the beginning.
        **kwargs (additional keyword arguments, *optional*):
            All other keyword arguments to pass to the regular `DataLoader` initialization.

    **Available attributes:**

        - **total_batch_size** (`int`) -- Total batch size of the dataloader across all processes.
            Equal to the original batch size when `split_batches=True`; otherwise the original batch size * the total
            number of processes

        - **total_dataset_length** (`int`) -- Total length of the inner dataset across all processes.
    """

    def __init__(
        self,
        dataset,
        device=None,
        rng_types=None,
        synchronized_generator=None,
        skip_batches=0,
        _drop_last: bool = False,
        _non_blocking: bool = False,
        **kwargs,
    ):
        super().__init__(dataset, **kwargs)
        self.device = device
        self.rng_types = rng_types
        self.synchronized_generator = synchronized_generator
        self.skip_batches = skip_batches
        self.gradient_state = GradientState()
        self._drop_last = _drop_last
        self._non_blocking = _non_blocking
        self.iteration = 0

    def __iter__(self):
        if self.rng_types is not None:
            synchronize_rng_states(self.rng_types, self.synchronized_generator)
        self.begin()

        self.set_epoch(self.iteration)
        dataloader_iter = super().__iter__()
        # We iterate one batch ahead to check when we are at the end
        try:
            current_batch = next(dataloader_iter)
        except StopIteration:
            yield

        batch_index = 0
        while True:
            try:
                # But we still move it to the device so it is done before `StopIteration` is reached
                if self.device is not None:
                    current_batch = send_to_device(current_batch, self.device, non_blocking=self._non_blocking)
                self._save_state_dict()
                next_batch = next(dataloader_iter)
                if batch_index >= self.skip_batches:
                    yield current_batch
                batch_index += 1
                current_batch = next_batch
            except StopIteration:
                self.end_of_dataloader = True
                if batch_index >= self.skip_batches:
                    yield current_batch
                break

        self.iteration += 1
        self.end()

    def set_epoch(self, epoch: int):
        # In case it is manually passed in, the user can set it to what they like
        if self.iteration != epoch:
            self.iteration = epoch
        if hasattr(self.batch_sampler, "sampler") and hasattr(self.batch_sampler.sampler, "set_epoch"):
            self.batch_sampler.sampler.set_epoch(epoch)
        # We support if a custom `Dataset` implementation has `set_epoch`
        # or in general HF datasets `Datasets`
        elif hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    @property
    def total_batch_size(self):
        batch_sampler = self.sampler if isinstance(self.sampler, BatchSampler) else self.batch_sampler
        return (
            batch_sampler.batch_size
            if getattr(batch_sampler, "split_batches", False)
            else (batch_sampler.batch_size * getattr(batch_sampler, "num_processes", 1))
        )

    @property
    def total_dataset_length(self):
        if hasattr(self.dataset, "total_length"):
            return self.dataset.total_length
        else:
            return len(self.dataset)
        
    @property
    def use_stateful_dataloader(self):
        return True

    def get_sampler(self):
        return get_sampler(self)

    def set_sampler(self, sampler):
        sampler_is_batch_sampler = isinstance(self.sampler, BatchSampler)
        if sampler_is_batch_sampler:
            self.sampler.sampler = sampler
        else:
            self.batch_sampler.sampler = sampler
            if hasattr(self.batch_sampler, "batch_sampler"):
                self.batch_sampler.batch_sampler.sampler = sampler

    def state_dict(self):
        return self.dl_state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.dl_state_dict = self.state_dict

    def _save_state_dict(self):
        self.dl_state_dict = super().state_dict()

class StatefulDataLoaderDispatcher(StatefulDataLoader, DataLoaderStateMixin):
    """
    Subclass of a torchdata `StatefulDataLoader` that will iterate and preprocess on process 0 only, then dispatch on each process
    their part of the batch.

    Args:
        split_batches (`bool`, *optional*, defaults to `False`):
            Whether the resulting `DataLoader` should split the batches of the original data loader across devices or
            yield full batches (in which case it will yield batches starting at the `process_index`-th and advancing of
            `num_processes` batches at each iteration). Another way to see this is that the observed batch size will be
            the same as the initial `dataloader` if this option is set to `True`, the batch size of the initial
            `dataloader` multiplied by `num_processes` otherwise. Setting this option to `True` requires that the batch
            size of the `dataloader` is a round multiple of `batch_size`.
        skip_batches (`int`, *optional*, defaults to 0):
            The number of batches to skip at the beginning of an iteration.

    **Available attributes:**

        - **total_batch_size** (`int`) -- Total batch size of the dataloader across all processes.
            Equal to the original batch size when `split_batches=True`; otherwise the original batch size * the total
            number of processes

        - **total_dataset_length** (`int`) -- Total length of the inner dataset across all processes.
    """

    def __init__(
        self,
        dataset,
        split_batches: bool = False,
        skip_batches=0,
        _drop_last: bool = False,
        _non_blocking: bool = False,
        slice_fn=None,
        **kwargs,
    ):
        shuffle = False
        if is_torch_version(">=", "1.11.0"):
            from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe

            # We need to save the shuffling state of the DataPipe
            if isinstance(dataset, ShufflerIterDataPipe):
                shuffle = dataset._shuffle_enabled
        super().__init__(dataset, **kwargs)
        self.split_batches = split_batches
        if shuffle:
            torch.utils.data.graph_settings.apply_shuffle_settings(dataset, shuffle=shuffle)

        self.gradient_state = GradientState()
        self.state = AcceleratorState()
        self._drop_last = _drop_last
        self._non_blocking = _non_blocking
        self.skip_batches = skip_batches

        self.slice_fn = slice_tensors if slice_fn is None else slice_fn
        self.iteration = 0

    def _fetch_batches(self, iterator):
        batches, batch = None, None
        # On process 0, we gather the batch to dispatch.
        if self.state.process_index == 0:
            try:
                if self.split_batches:
                    # One batch of the main iterator is dispatched and split.
                    self._save_state_dict()
                    batch = next(iterator)
                else:
                    # num_processes batches of the main iterator are concatenated then dispatched and split.
                    # We add the batches one by one so we have the remainder available when drop_last=False.
                    batches = []
                    for _ in range(self.state.num_processes):
                        self._save_state_dict()
                        batches.append(next(iterator))
                    try:
                        batch = concatenate(batches, dim=0)
                    except RuntimeError as e:
                        raise RuntimeError(
                            "You can't use batches of different size with `dispatch_batches=True` or when using an `IterableDataset`."
                            "either pass `dispatch_batches=False` and have each process fetch its own batch "
                            " or pass `split_batches=True`. By doing so, the main process will fetch a full batch and "
                            "slice it into `num_processes` batches for each process."
                        ) from e
                # In both cases, we need to get the structure of the batch that we will broadcast on other
                # processes to initialize the tensors with the right shape.
                # data_structure, stop_iteration
                batch_info = [get_data_structure(batch), False]
            except StopIteration:
                batch_info = [None, True]
        else:
            batch_info = [None, self._stop_iteration]
        # This is inplace, so after this instruction, every process has the same `batch_info` as process 0.
        broadcast_object_list(batch_info)
        self._stop_iteration = batch_info[1]
        if self._stop_iteration:
            # If drop_last is False and split_batches is False, we may have a remainder to take care of.
            if not self.split_batches and not self._drop_last:
                if self.state.process_index == 0 and len(batches) > 0:
                    batch = concatenate(batches, dim=0)
                    batch_info = [get_data_structure(batch), False]
                else:
                    batch_info = [None, True]
                broadcast_object_list(batch_info)
        return batch, batch_info

    def __iter__(self):
        self.begin()
        self.set_epoch(self.iteration)
        main_iterator = None
        if is_torch_version(">=", "2.0.1"):
            # NOTE PyTorch DataLoader adds forward compatibilities for DataPipes, which broadcasts
            # shared seed to all dist processes. Thus, we need to create iterator for all dist processes.
            # But, we only iterate through the DataLoader on process 0.
            main_iterator = super().__iter__()
        elif self.state.process_index == 0:
            main_iterator = super().__iter__()
        stop_iteration = False
        self._stop_iteration = False
        first_batch = None
        next_batch, next_batch_info = self._fetch_batches(main_iterator)
        batch_index = 0
        while not stop_iteration:
            batch, batch_info = next_batch, next_batch_info

            if self.state.process_index != 0:
                # Initialize tensors on other processes than process 0.
                batch = initialize_tensors(batch_info[0])
            batch = send_to_device(batch, self.state.device, non_blocking=self._non_blocking)
            # Broadcast the batch before splitting it.
            batch = broadcast(batch, from_process=0)

            if not self._drop_last and first_batch is None:
                # We keep at least num processes elements of the first batch to be able to complete the last batch
                first_batch = self.slice_fn(
                    batch,
                    slice(0, self.state.num_processes),
                    process_index=self.state.process_index,
                    num_processes=self.state.num_processes,
                )

            if batch is None:
                raise ValueError(
                    f"Batch does not contain any data (`{batch}`). At the end of all iterable data available before expected stop iteration."
                )

            observed_batch_size = find_batch_size(batch)
            batch_size = observed_batch_size // self.state.num_processes

            stop_iteration = self._stop_iteration
            if not stop_iteration:
                # We may still be at the end of the dataloader without knowing it yet: if there is nothing left in
                # the dataloader since the number of batches is a round multiple of the number of processes.
                next_batch, next_batch_info = self._fetch_batches(main_iterator)
                # next_batch_info[0] is None when there are no more batches, otherwise we still need to process them.
                if self._stop_iteration and next_batch_info[0] is None:
                    stop_iteration = True

            if not self._drop_last and stop_iteration and observed_batch_size % self.state.num_processes != 0:
                # If the last batch is not complete, let's add the first batch to it.
                batch = concatenate([batch, first_batch], dim=0)
                # Batch size computation above is wrong, it's off by 1 so we fix it.
                batch_size += 1

            data_slice = slice(self.state.process_index * batch_size, (self.state.process_index + 1) * batch_size)
            batch = self.slice_fn(
                batch,
                data_slice,
                process_index=self.state.process_index,
                num_processes=self.state.num_processes,
            )

            if stop_iteration:
                self.end_of_dataloader = True
                self.remainder = observed_batch_size
            if batch_index >= self.skip_batches:
                yield batch
            batch_index += 1
        self.iteration += 1
        self.end()

    def set_epoch(self, epoch: int):
        # In case it is manually passed in, the user can set it to what they like
        if self.iteration != epoch:
            self.iteration = epoch
        if hasattr(self.batch_sampler.sampler, "set_epoch"):
            self.batch_sampler.sampler.set_epoch(epoch)
        elif hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def __len__(self):
        whole_length = super().__len__()
        if self.split_batches:
            return whole_length
        elif self._drop_last:
            return whole_length // self.state.num_processes
        else:
            return math.ceil(whole_length / self.state.num_processes)

    @property
    def total_batch_size(self):
        return (
            self.dataset.batch_size if self.split_batches else (self.dataset.batch_size * self.dataset.num_processes)
        )

    @property
    def total_dataset_length(self):
        return len(self.dataset)

    @property
    def use_stateful_dataloader(self):
        return True

    def get_sampler(self):
        return get_sampler(self)

    def set_sampler(self, sampler):
        sampler_is_batch_sampler = isinstance(self.sampler, BatchSampler)
        if sampler_is_batch_sampler:
            self.sampler.sampler = sampler
        else:
            self.batch_sampler.sampler = sampler
            if hasattr(self.batch_sampler, "batch_sampler"):
                self.batch_sampler.batch_sampler.sampler = sampler

    def state_dict(self):
        return self.dl_state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.dl_state_dict = self.state_dict

    def _save_state_dict(self):
        self.dl_state_dict = super().state_dict()

class StatefulSkipDataLoader(StatefulDataLoader):
    """
    Subclass of a torchdata `StatefulDataLoader` that will skip the first batches.

    Args:
        dataset (`torch.utils.data.dataset.Dataset`):
            The dataset to use to build this datalaoder.
        skip_batches (`int`, *optional*, defaults to 0):
            The number of batches to skip at the beginning.
        kwargs:
            All other keyword arguments to pass to the regular `DataLoader` initialization.
    """

    def __init__(self, dataset, skip_batches=0, **kwargs):
        super().__init__(dataset, **kwargs)
        self.skip_batches = skip_batches

    def __iter__(self):
        for index, batch in enumerate(super().__iter__()):
            if index >= self.skip_batches:
                self._save_state_dict()
                yield batch

    @property
    def use_stateful_dataloader(self):
        return True

    def state_dict(self):
        return self.dl_state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.dl_state_dict = self.state_dict

    def _save_state_dict(self):
        self.dl_state_dict = super().state_dict()