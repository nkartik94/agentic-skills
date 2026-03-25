---
name: production-git
description: "Git branching strategy, commit conventions, PR workflow, release management, hotfix procedures, rollback, and tagging for production software projects. Use this skill when creating branches, writing commit messages, opening pull requests, preparing releases, handling production incidents, or reviewing git history and workflow. Triggers on any git-related task: branching, committing, merging, tagging, or deployment workflow."
license: MIT
metadata:
  author: nkartik94
  version: "1.0.0"
---

# Production Git

Apply every rule below whenever branching, committing, merging, releasing, or deploying code.

## When to Apply

- Creating a new branch for a feature, bug fix, or refactor
- Writing commit messages
- Opening or reviewing pull requests
- Preparing a release or deployment
- Handling production incidents requiring hotfixes or rollbacks
- Tagging versions or managing semver
- Reviewing or advising on git workflow

---

## Quick Reference

| Rule | Pattern |
|------|---------|
| Main branch | `main` — never push directly |
| Feature branch | `feat/<description>` |
| Bug fix branch | `fix/<description>` |
| Hotfix branch | `hotfix/<description>` |
| Refactor branch | `refactor/<description>` |
| Release branch | `release/v<semver>` |
| Commit format | `<type>: <short description>` (Conventional Commits) |
| Merge strategy | Squash and merge (default for most PRs) |
| Tags | `v<MAJOR>.<MINOR>.<PATCH>` — always annotated |
| Versioning | Semantic Versioning — MAJOR.MINOR.PATCH |
| Branch off from | `main` (features/fixes) or `release/*` (hotfixes only) |
| Merge back to | `main` via PR; release branches also merge back to `main` after deploy |

---

## 1. Branch Naming

### Prefixes

| Prefix | Use Case | Branches From | Merges Into |
|--------|----------|---------------|-------------|
| `feat/` | New feature or capability | `main` | `main` via PR |
| `fix/` | Bug fix (non-urgent) | `main` | `main` via PR |
| `hotfix/` | Urgent production fix | `release/*` | `release/*` via PR, then back to `main` |
| `refactor/` | Code restructure, no behavior change | `main` | `main` via PR |
| `chore/` | Dependencies, config, tooling | `main` | `main` via PR |
| `docs/` | Documentation only | `main` | `main` via PR |
| `release/` | Release candidate / stable snapshot | `main` | Back to `main` after deploy |

### Naming Rules

- Lowercase and hyphens only: `fix/api-null-response`
- 3–5 words — describe **what changed**, not that something changed
- No version numbers on feature/fix branches — versions belong on `release/*` and tags
- No author names — git tracks authorship already
- No dates — git tracks timestamps already

### Good vs Bad

| Good | Bad | Why Bad |
|------|-----|---------|
| `feat/multi-sheet-extraction` | `feat/new-feature` | Too vague |
| `fix/pdf-null-check` | `fix/bug` | Says nothing specific |
| `refactor/extraction-pipeline` | `refactor/v2` | Version is not a description |
| `release/v0.33.0` | `release/march-update` | Use semver, not dates |
| `chore/update-dependencies` | `update-stuff` | Missing prefix, too vague |

---

## 2. Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short description>
```

### Commit Types

| Type | When | Example |
|------|------|---------|
| `feat` | New feature | `feat: add multi-sheet extraction support` |
| `fix` | Bug fix | `fix: null check on API response` |
| `refactor` | Code restructure (no behavior change) | `refactor: simplify extraction pipeline` |
| `chore` | Dependencies, config, CI | `chore: update dependencies to latest` |
| `docs` | Documentation only | `docs: add deployment guide` |
| `perf` | Performance improvement | `perf: optimize chunk processing` |
| `test` | Adding or updating tests | `test: add unit tests for field mapper` |
| `ci` | CI/CD pipeline changes | `ci: add GitHub Actions workflow` |
| `build` | Build system / Dockerfile | `build: update base image to python 3.12` |
| `revert` | Reverting a previous commit | `revert: undo field mapper changes` |

### Multi-line Format (for complex changes)

```
feat: add csv file format support

- Add new parser for CSV input files
- Register csv processor in the file type registry
- Update file type configuration YAML

Closes #42
```

Rules:
- First line: 72 chars max, imperative mood ("add", not "added")
- Blank line between subject and body
- Body explains *what* and *why*, not *how*
- Reference issue/ticket numbers in footer

---

## 3. Day-to-Day Workflow

Every piece of work follows this sequence:

```bash
# 1. Start from latest main
git checkout main
git pull origin main

# 2. Create a branch
git checkout -b feat/your-feature-name

# 3. Make changes, test locally

# 4. Stage and commit
git add .
git commit -m "feat: add your feature description"

# 5. Push branch
git push origin feat/your-feature-name

# 6. Open PR on GitHub → main

# 7. Squash and merge, delete branch
```

### Switching Tasks Mid-Work

Use `git stash` to shelve uncommitted changes when you need to switch branches:

```bash
git stash                          # Save current work aside
git checkout main
git checkout -b fix/urgent-bug     # Work on something else
# ... fix, commit, push, PR ...
git checkout feat/your-feature-name
git stash pop                      # Restore shelved work
```

---

## 4. Pull Request Procedure

### Title

Use commit message format: `feat: add multi-sheet extraction support`

### Description Template

```markdown
## Summary
- <what changed and why>
- <bullet points>

## Testing
- [ ] Tested locally
- [ ] Verified expected behavior
- [ ] No regressions

## Breaking Changes
- None / <list any>
```

### Merge Strategy

| Strategy | What It Does | When to Use |
|----------|-------------|-------------|
| **Squash and merge** | Collapses all branch commits into one on main | **Default** — keeps history clean |
| **Merge commit** | Preserves all commits + adds merge commit | Long-running feature where commits tell a meaningful story |
| **Rebase and merge** | Replays commits on top of main, no merge commit | Small PRs with already-clean commit history |

Default to **squash and merge** — most branch commits are incremental noise (`fix typo`, `forgot file`). One commit per PR = clean `git log`, easy to revert.

Rules:
- Delete branch after merging
- PR title becomes the squash commit message — make it descriptive
- Never merge your own PR without review on team projects

---

## 5. Release Procedure

### When to Cut a Release

- A set of features/fixes is ready for production
- You want a stable, deployable, tagged snapshot
- A code freeze is needed before deployment

### Steps

```bash
# 1. Ensure main is up to date
git checkout main
git pull origin main

# 2. Create release branch
git checkout -b release/v0.33.0
git push origin release/v0.33.0

# 3. Only bugfixes go in from here — branch FROM the release branch
git checkout -b fix/release-bug release/v0.33.0
# ... fix, commit, PR back into release/v0.33.0 ...

# 4. After all fixes are tested, tag it
git checkout release/v0.33.0
git pull origin release/v0.33.0
git tag -a v0.33.0 -m "Release v0.33.0 — brief description"
git push origin v0.33.0

# 5. Merge release branch back into main
git checkout main
git pull origin main
git merge release/v0.33.0
git push origin main
```

### Semantic Versioning

```
v<MAJOR>.<MINOR>.<PATCH>
```

| Component | When to Bump | Example |
|-----------|-------------|---------|
| MAJOR | Breaking changes | `v1.0.0` → `v2.0.0` |
| MINOR | New features, backward compatible | `v0.32.0` → `v0.33.0` |
| PATCH | Bug fixes only | `v0.33.0` → `v0.33.1` |

---

## 6. Hotfix Procedure

When production breaks after a release:

```bash
# 1. Branch FROM the release branch (not main!)
git checkout release/v0.33.0
git pull origin release/v0.33.0
git checkout -b fix/critical-production-bug

# 2. Fix, commit
git add .
git commit -m "fix: description of critical fix"
git push origin fix/critical-production-bug

# 3. PR into the release branch (not main)
# Open PR: fix/critical-production-bug → release/v0.33.0

# 4. Tag the hotfix
git checkout release/v0.33.0
git pull origin release/v0.33.0
git tag -a v0.33.1 -m "Hotfix v0.33.1 — description"
git push origin v0.33.1

# 5. Deploy the hotfix tag

# 6. Merge release branch back into main
git checkout main
git pull origin main
git merge release/v0.33.0
git push origin main
```

### Hotfix When Next Release Is Already in Progress

If `release/v0.34.0` is being worked on while `v0.33.0` needs a hotfix:

1. Fix on `release/v0.33.0` → tag `v0.33.1`
2. Merge fix into `release/v0.34.0` first
3. Then merge `release/v0.34.0` into `main`

This ensures the fix propagates to all active branches.

---

## 7. Rollback Procedure

### Rollback to Previous Tag

```bash
# On your server / deployment environment:
git fetch --all --tags
git tag -l                    # List available tags
git checkout v0.32.2          # Checkout last known good version
# Rebuild and redeploy
```

### Emergency Sequence

```bash
git fetch --all --tags
git checkout v0.32.2          # Last known working version
# Full clean rebuild
# Watch logs to confirm stability
```

### Check What's Currently Deployed

```bash
git log --oneline -1          # Current commit
git describe --tags           # Current tag (if on a tag)
```

---

## 8. Tagging & Versioning

```bash
# Create annotated tag (always use annotated — includes author, date, message)
git tag -a v0.33.0 -m "Release v0.33.0 — multi-sheet support"
git push origin v0.33.0       # Push tags separately from commits

# List tags
git tag -l
git tag -l -n1                # With messages

# Delete a tag (if created by mistake)
git tag -d v0.33.0            # Local
git push origin :v0.33.0      # Remote
```

Rules:
- Always prefix with `v`: `v0.33.0` not `0.33.0`
- Always use **annotated** tags (`-a`), never lightweight
- Always tag on the release branch, never on main directly
- Push tags explicitly — `git push` does not push tags by default
- Hotfix tags increment PATCH: `v0.33.0` → `v0.33.1`

---

## 9. Anti-Patterns

| Anti-Pattern | Why It's Bad | Do This Instead |
|-------------|-------------|-----------------|
| Push directly to `main` | No review, no rollback point | Always branch + PR |
| `git push --force` on shared branches | Destroys others' work | Only force-push on your own unshared feature branches |
| Skip tagging releases | Can't roll back to exact version | Always tag before deploying |
| Commit `.env` files | Exposes secrets in git history | `.gitignore` the `.env`, commit `.env.example` |
| Work directly on release branches | Messy history, hard to track | Branch from release, PR back in |
| Forget to merge release back to main | Main drifts from production | Always merge release → main after deploy |
| Long-lived feature branches (weeks/months) | Merge conflicts, drift from main | Keep branches short-lived, merge frequently |
| Name branches with dates or author names | Not descriptive, clutters list | Use `prefix/what-changed` pattern |
| Vague branch names (`fix/bug`, `feat/new`) | Nobody knows what it is | Describe the actual change in 3–5 words |
| Multiple local copies instead of branches | No history, no diff, fragile | Git branches serve exactly this purpose |
| `git commit -m "wip"` as final commit | Meaningless history | Squash messy commits before PR, write a proper message |
| `Co-Authored-By: <LLM name>` in commits | LLMs are tools, not contributors — clutters history and misrepresents authorship | Omit entirely; you wrote the code, you own the commit |

---

## 10. Pre-Merge Checklist

- [ ] Branch name follows `prefix/description` format
- [ ] Commit messages follow Conventional Commits (`type: description`)
- [ ] PR title is descriptive and matches commit format
- [ ] No direct push to `main`
- [ ] Tests pass locally
- [ ] No `.env` or secrets committed
- [ ] Release branch tagged before deploying
- [ ] Release branch merged back to `main` after deploy

---

For detailed explanations of Git core concepts, merge strategy deep-dives, industry branching model comparisons, scenario recipes, and deployment patterns, see [references/REFERENCE.md](references/REFERENCE.md).
