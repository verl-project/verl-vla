# Tests layout

Each folder under `tests/` corresponds to a test category for a sub-namespace in `verl_vla`.
For instance:

- `tests/trainer` for testing functionality related to `verl_vla/trainer`
- `tests/models` for testing functionality related to `verl_vla/models`
- `tests/envs` for testing functionality related to `verl_vla/envs`

There is also a `special_sanity` folder for quick repository checks that are useful in local development
and pre-commit.

# Workflow layout

For this lightweight repository, the default workflow is:

1. Run repository sanity checks via `pre-commit run --all-files`
2. Run unit tests via `pytest`

As the repository grows, place tests under `tests/<module_name>/...` so the structure mirrors `src/verl_vla/...`.
