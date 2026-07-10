### What does this PR do?

> Add a concise overview of what this PR aims to achieve. Reference related issues and PRs that help with review.

### Checklist Before Starting

- [ ] Search for similar PRs. Paste at least one query link here: ...
- [ ] Format the PR title as `[{modules}] {type}: {description}`. CI checks this format.
  - `{modules}` include `trainer`, `rollout`, `worker`, `env`, `model`, `data`, `teleop`, `recorder`, `entrypoints`, `cfg`, `docker`, `ci`, `doc`, `perf`, and `misc`.
  - Separate multiple modules with a comma and one space, for example `[env, recorder]`.
  - `{type}` is one of `feat`, `fix`, `refactor`, `chore`, or `test`.
  - Add `[BREAKING]` at the beginning when the PR breaks an API, CLI argument, configuration, or function signature.
  - Use `[1/N]` at the beginning only when the change is intentionally split across a related PR series.
  - Example: `[BREAKING][env, cfg] refactor: rename simulator configuration fields`.

### Test

> List the exact commands and experiments run. If GPU, LIBERO, Isaac, or LeRobot validation was not feasible, explain why and state what remains unverified.

### API and Usage Example

> Demonstrate user-visible API, CLI, or configuration changes. Write `N/A` when there is no user-visible change.

```python
# Add a minimal usage example when applicable.
```

### Design & Code Changes

> Describe the high-level design for complex changes and list the important implementation decisions.

### Checklist Before Submitting

> [!IMPORTANT]
> Check every applicable item before requesting review.

- [ ] Read the [Contributing Guide](https://github.com/verl-project/verl-vla/blob/main/CONTRIBUTING.md).
- [ ] Run `pre-commit run --all-files --show-diff-on-failure --color=always`.
- [ ] Add or update documentation for user-visible behavior, or explain why documentation is not applicable.
- [ ] Add or update tests, or explain why automated coverage is not feasible.
- [ ] Disclose AI assistance when applicable.
