# Contributing to verl-vla

## Commit Messages

Commit titles use the same module-first style as `verl`:

```text
[module] type: description
[module, module] type: description
[BREAKING][module] type: description
```

Examples:

```text
[teleop] feat: add gamepad calibration
[env, recorder] fix: preserve terminal signals across action chunks
[BREAKING][cfg] refactor: rename rollout configuration fields
```

Allowed types are:

- `feat`: add or extend functionality
- `fix`: correct incorrect behavior
- `refactor`: restructure code without changing its behavior
- `chore`: perform maintenance work
- `test`: add or update tests

Allowed modules are:

```text
trainer rollout worker env model data teleop recorder entrypoints cfg
docker ci doc perf misc
```

Follow these title rules:

- Use lowercase module and type names.
- Separate multiple modules with a comma followed by one space.
- Write the description in English, using an imperative verb with a lowercase first letter.
- Do not end the description with a period.
- Keep the title at or below 100 characters.
- Prefix an incompatible API, CLI, or configuration change with `[BREAKING]`.

Autosquash titles such as `fixup!`, `squash!`, and `amend!` are accepted when
the title they reference follows this convention. Git-generated merge and
revert titles are also accepted.

Install the repository hooks after cloning:

```bash
pip install pre-commit
pre-commit install
```

## Pull Requests

Pull request titles follow the commit title convention above. A related PR
series may add a leading progress marker such as `[1/N]`:

```text
[env] fix: preserve terminal signals across action chunks
[1/N][trainer] feat: add a staged training pipeline
[BREAKING][cfg] refactor: rename rollout configuration fields
```

Before opening or updating a pull request:

- Search for existing pull requests that address the same change.
- Fill out every applicable section of the pull request template.
- Run `pre-commit run --all-files --show-diff-on-failure --color=always`.
- Add or update tests, or explain why automated coverage is not feasible.
- Report the exact checks and experiments run. For GPU, LIBERO, Isaac, or
  LeRobot validation that cannot run locally, state what remains unverified.
- Document user-visible API, CLI, and configuration changes.
- Disclose AI assistance when applicable.
