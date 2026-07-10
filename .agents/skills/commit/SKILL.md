---
name: commit
description: Prepare and create a local Git commit following verl-vla conventions. Use when the user asks to commit current changes, organize a coherent local commit, or validate a proposed commit message; require final confirmation before staging or committing, and do not push.
---

Create a commit only when the user explicitly asks for one. Do not push,
rebase, squash, or amend unless the user separately requests that operation.

## Gather Context

Read the commit message rules in the repository-root `CONTRIBUTING.md`, then
inspect the repository state and both staged and unstaged changes:

```bash
git status --short
git diff
git diff --cached
```

Use the repository convention as the source of truth for allowed modules and
types. Check recent commit titles when useful, but do not copy an old title
that violates the current convention.

## Define the Commit Scope

Identify one coherent change to commit. Preserve unrelated user changes and
pre-existing staged content. If staged content cannot be safely separated
from the requested change, stop and explain the conflict instead of modifying
the user's staging area without permission.

Choose modules by responsibility rather than implementation detail. Use the
smallest set that accurately describes the change, and explain the specific
algorithm, backend, or dependency in the description when needed.

## Validate the Change

Run checks relevant to the proposed change. Do not claim a check passed unless
it was actually run. Keep the staging area unchanged while preparing the
commit preview.

## Compose and Validate the Message

Write a concise title that follows `CONTRIBUTING.md`. Add a body when the
reason for the change, an important tradeoff, or migration guidance would not
be clear from the title and diff alone.

Validate the proposed title before creating the commit:

```bash
python3 scripts/check_commit_message.py --title "[module] type: description"
```

If validation fails, fix the message. Never bypass repository hooks with
`--no-verify`.

## Request Final Confirmation

Before staging or committing, show the user:

- The proposed title and body, if any.
- The explicit list of files to include.
- Checks run and their results.
- Any unrelated or pre-existing staged changes that will remain untouched.
- A statement that the commit will remain local and will not be pushed.

Wait for explicit confirmation such as "confirm", "commit", or "go ahead".
The initial request to prepare a commit starts this workflow but does not
replace final confirmation of the preview. Do not run `git add` or `git commit`
before that confirmation.

If any candidate file changes after the preview, refresh the diff and request
confirmation again.

## Commit and Report

After confirmation, stage only the approved files with
`git add -- <path>...`. Do not use `git add .` or `git add -A`, because they
can capture unrelated work.

Inspect the final staged diff and check it for whitespace errors:

```bash
git diff --cached --check
git diff --cached
```

Create the local commit, then report its hash, title, checks run, and remaining
worktree state. Do not push the commit.
