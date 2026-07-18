# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
import logging

import numpy as np
import torch
from tqdm import tqdm

from gr00t.configs.base_config import Config
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
from gr00t.data.dataset.sharded_single_step_dataset import (
    ShardedSingleStepDataset,
    extract_step_data,
)
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.stats import generate_rel_stats, generate_stats
from gr00t.data.types import MessageType
from gr00t.utils.dist_utils import run_or_wait_on_rank0


class InMemoryValDataset(torch.utils.data.Dataset):
    """Fixed, fully preprocessed validation samples for periodic eval_loss."""

    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class DatasetFactory:
    """
    Factory class for building training datasets. Model-agnostic.
    """

    def __init__(self, config: Config):
        self.config = config

    def build(
        self, processor: BaseProcessor
    ) -> tuple[ShardedMixtureDataset, ShardedMixtureDataset | None]:
        """Build the dataset. Returns a tuple of (train_dataset, eval_dataset).

        The sharded train pipeline has no streaming eval support; the eval dataset
        (when any dataset spec sets ``val_dataset_path``) is a small fixed set of
        fully preprocessed timesteps evaluated with the standard HF eval loop.
        """
        if self.config.training.eval_strategy != "no" and not any(
            spec.val_dataset_path for spec in self.config.data.datasets
        ):
            raise ValueError(
                "eval_strategy requires at least one dataset with val_dataset_path "
                "(the sharded train dataset does not support evaluation)"
            )

        all_datasets = []
        all_weights = []
        for dataset_spec in tqdm(
            self.config.data.datasets,
            total=len(self.config.data.datasets),
            desc="Initializing datasets",
        ):
            datasets = []
            for dataset_path in dataset_spec.dataset_paths:
                embodiment_tag = dataset_spec.embodiment_tag
                assert embodiment_tag is not None, "Embodiment tag is required"
                assert self.config.data.mode == "single_turn", "Only single turn mode is supported"
                # rank-0 writes stats; helper barriers before peers read them.
                with run_or_wait_on_rank0(label=f"generate_stats({dataset_path})") as is_rank0:
                    if is_rank0:
                        generate_stats(dataset_path)
                        generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
                dataset = ShardedSingleStepDataset(
                    dataset_path=dataset_path,
                    embodiment_tag=EmbodimentTag(embodiment_tag),
                    modality_configs=self.config.data.modality_configs[embodiment_tag],
                    shard_size=self.config.data.shard_size,
                    episode_sampling_rate=self.config.data.episode_sampling_rate,
                    seed=self.config.data.seed,
                    allow_padding=self.config.data.allow_padding,
                )
                datasets.append(dataset)
            dataset_lengths = np.array([len(dataset) for dataset in datasets])
            dataset_relative_lengths = dataset_lengths / dataset_lengths.sum()
            for dataset, relative_length in zip(datasets, dataset_relative_lengths):
                weight = relative_length * dataset_spec.mix_ratio
                all_datasets.append(dataset)
                all_weights.append(weight)

        alpha = self.config.data.ds_weights_alpha
        if alpha is not None and len(all_datasets) > 1:
            ds_lengths = np.array([len(dataset) for dataset in all_datasets], dtype=np.float64)
            all_weights = (np.power(ds_lengths, alpha) / np.power(ds_lengths[0], alpha)).tolist()
            print(
                f"Applied ds_weights_alpha={alpha} across {len(all_datasets)} datasets; "
                "this overrides per-dataset mix_ratio sampling weights."
            )

        train_dataset = ShardedMixtureDataset(
            datasets=all_datasets,
            weights=all_weights,
            processor=processor,
            seed=self.config.data.seed,
            training=True,
            num_shards_per_epoch=self.config.data.num_shards_per_epoch,
            override_pretraining_statistics=self.config.data.override_pretraining_statistics,
        )
        # Must come after ShardedMixtureDataset init: merge_statistics() configures
        # the processor with the train-split statistics the val samples must use.
        eval_dataset = self._build_val_dataset(processor)
        return train_dataset, eval_dataset

    def _build_val_dataset(self, processor: BaseProcessor) -> InMemoryValDataset | None:
        """Preprocess a fixed, evenly spaced subset of each configured val split.

        Samples are built with a deepcopied processor in eval mode (deterministic
        image transform, no state dropout) so eval_loss is comparable across steps.
        """
        specs = [spec for spec in self.config.data.datasets if spec.val_dataset_path]
        if not specs:
            return None

        eval_processor = deepcopy(processor)
        eval_processor.eval()

        samples = []
        budget_per_ds = max(1, self.config.data.val_max_samples // len(specs))
        for spec in specs:
            embodiment_tag = EmbodimentTag(spec.embodiment_tag)
            modality_configs = self.config.data.modality_configs[spec.embodiment_tag]
            loader = LeRobotEpisodeLoader(
                dataset_path=spec.val_dataset_path,
                modality_configs=modality_configs,
            )
            action_deltas = modality_configs["action"].delta_indices
            horizon = max(action_deltas) - min(action_deltas) + 1
            num_episodes = len(loader)
            per_episode = max(1, -(-budget_per_ds // num_episodes))  # ceil division
            for ep in range(num_episodes):
                episode = loader[ep]
                last_start = len(episode) - horizon
                if last_start < 0:
                    continue
                step_indices = np.unique(
                    np.linspace(0, last_start, num=min(per_episode, last_start + 1), dtype=int)
                )
                for step in step_indices:
                    vla_step_data = extract_step_data(
                        episode, int(step), modality_configs, embodiment_tag
                    )
                    messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
                    samples.append(eval_processor(messages))
                del episode
            logging.info("Built %d validation samples from %s", len(samples), spec.val_dataset_path)
        return InMemoryValDataset(samples)
