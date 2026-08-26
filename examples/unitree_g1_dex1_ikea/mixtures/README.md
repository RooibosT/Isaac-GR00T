# Mixture specs

Each file lists the datasets to train *alongside* the one passed as
`--dataset-path`; that primary dataset carries its own weight through
`--mix-ratio`. Sampling weight is `relative_length x mix_ratio`
(`gr00t/data/dataset/factory.py`), so leaving every ratio at 1.0 reproduces the
natural proportions and nothing is ever dropped -- the ratio only changes how
often a dataset's shards come up.

| file | what it adds |
|---|---|
| `ikea_plus_new_tasks.json` | the rotate-tabletop and flip episodes, so the older three tasks can be weighted against them from the command line |
| `ikea_plus_sim.json` | the lightwheel simulator, on its own embodiment tag (rejected, see EXPERIMENTS.md section 15) |
| `ikea_plus_sim_plus_bct.json` | the above plus the BCT recordings |

## Weighting the older three tasks

`insert` lost 12.1% on EE when the two new tasks joined the mixture, and it was
also down to 16.4 epochs against the 19.6 a model trained on the older three
alone received. To put those epochs back, weight the primary:

    DATASET_ROOT=.../G1_Dex1_IKEA_table_30hz_v2 \
    MIXTURE=.../mixtures/ikea_plus_new_tasks.json MIX_RATIO=1.6 ...

| `--mix-ratio` on the older set | its share | epochs it gets in 30k |
|---:|---:|---:|
| 1.0 (natural) | 55.8% | 16.4 |
| **1.6** | 66.9% | **19.6** -- matches the old-tasks-only model |
| 3.0 | 79.1% | 23.2 |

Note this changes two things at once: how often the older episodes are sampled,
and the merged normalisation statistics, which `merge_statistics` weights by the
same sampling weight.
