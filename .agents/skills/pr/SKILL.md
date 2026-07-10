---
name: pr
description: Prepare, create, or update a verl-vla pull request following repository conventions. Use when the user asks to open, draft, publish, or revise a PR; inspect the complete branch diff and existing GitHub edits, require final confirmation before creating a branch, pushing, or changing a PR.
---

Do not create branches, push commits, or create or update a pull request until
the user confirms the final preview. Never force-push unless the user
explicitly requests it.

## Gather Context

Read the repository-root `CONTRIBUTING.md` and
`.github/PULL_REQUEST_TEMPLATE.md`. Inspect:

```bash
git status --short
git branch --show-current
git remote -v
git log --oneline origin/main..HEAD
git diff --stat origin/main...HEAD
git diff origin/main...HEAD
```

Determine the intended base branch; default to `main` only when the user did
not specify another base. Include the complete branch diff, not just the most
recent commit.

Search open pull requests for the same issue or change. If the current branch
already has a PR, read its current title, body, draft state, review comments,
and requested changes before proposing an update. Preserve edits made directly
on GitHub instead of rebuilding the body from scratch.

If the current branch is `main`, propose a focused branch name. Do not create
that branch before final confirmation.

## Validate the Change

Run checks relevant to the complete PR and record exact results. Run the
repository pre-submit check unless the user explicitly narrows validation:

```bash
pre-commit run --all-files --show-diff-on-failure --color=always
```

Do not claim GPU, LIBERO, Isaac, or LeRobot validation ran when the environment
did not support it. State what remains unverified.

## Compose the PR

Create a title that follows `CONTRIBUTING.md`, including `[BREAKING]` or an
optional PR-series prefix when applicable. Validate it with:

```bash
python3 scripts/check_pr_title.py --title "[module] type: description"
```

Fill every applicable section of `.github/PULL_REQUEST_TEMPLATE.md`. Only
check checklist items that were actually completed. Make the overview concise,
list exact validation commands, explain missing coverage, and include API or
usage examples for user-visible changes.

When updating an existing PR, ensure the title and body describe all branch
changes while retaining useful user-authored content.

## Request Final Confirmation

Before any branch creation, push, PR creation, or PR update, show the user:

- Base and head branches, including a proposed branch name when needed.
- Commits and files included in the complete PR diff.
- Proposed title and full body.
- Checks run, results, and anything unverified.
- Whether the PR will be draft or ready for review.
- Exact external actions to perform, including the push destination.

Wait for explicit confirmation. The initial request starts preparation but
does not replace confirmation of this preview. If the branch, commits, files,
title, body, or remote PR changes after the preview, refresh it and request
confirmation again.

## Create or Update the PR

After confirmation, recheck the branch and worktree, then perform only the
approved actions. Push without force. Create the PR against the approved base,
or update the existing PR without overwriting concurrent GitHub edits.

Return the PR URL, final title, base and head branches, draft state, checks
run, and remaining worktree state.
