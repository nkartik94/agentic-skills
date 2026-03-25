# Production Git — Extended Reference

Detailed explanations, industry context, scenario recipes, and deployment patterns for the production-git skill.
See [SKILL.md](../SKILL.md) for the core rules.

---

## Git Core Concepts

### How Code Gets from Your Machine to Production

```
1. Edit files locally
        ↓
2. git add .              → Stage: select which changes to include
        ↓
3. git commit -m "msg"    → Commit: save a snapshot LOCALLY (no one else sees it)
        ↓
4. git push               → Push: upload commits to GitHub (visible on YOUR branch only)
        ↓
5. Open PR + merge        → Merge: put your branch's commits into main
        ↓
6. Deploy                 → Pull latest on server, rebuild, restart
```

### Commit vs Push vs Merge

| Action | What It Does | Who Can See It |
|--------|-------------|----------------|
| `git commit` | Saves a snapshot of changes **locally** | Only you |
| `git push` | Uploads local commits to **GitHub** | Anyone viewing your branch on GitHub |
| Merge (via PR) | Combines your branch's commits **into main** | Everyone — it's now part of the shared codebase |

**Key distinctions:**
- After `commit` — saved locally, GitHub doesn't know about it
- After `push` — on GitHub, but only on your branch (main is untouched)
- After `merge` — now part of main, ready to deploy

You can make **multiple commits before pushing**:
```bash
git commit -m "add new file"        # saved locally
git commit -m "fix typo in file"    # saved locally
git commit -m "update config"       # saved locally
git push                            # all 3 commits uploaded at once
```

### Push vs Pull — Syncing with GitHub

```
git push  →  your machine ──→ GitHub    (upload)
git pull  →  GitHub ──→ your machine    (download)
```

**`origin`** is a nickname for your GitHub repo URL. Verify with `git remote -v`.

| Command | What It Does |
|---------|-------------|
| `git push origin main` | Push to GitHub's `main` branch |
| `git push origin feat/new-feature` | Push to GitHub's feature branch |
| `git push` | Shorthand — push to tracked branch (works after first explicit push) |
| `git pull origin main` | Download latest from GitHub's `main` |
| `git pull` | Shorthand — pull from tracked remote |

**First push of a new branch must be explicit:**
```bash
git checkout -b feat/new-feature
git push origin feat/new-feature    # first time: must specify
git push                            # subsequent: shorthand works
```

### git stash — Shelve Work Temporarily

```bash
git stash                    # Save uncommitted changes aside
git stash pop                # Restore the most recent stash
git stash list               # Show all saved stashes
git stash drop               # Discard the most recent stash
```

**When to use:** Switching branches when you have uncommitted work, or moving changes to a different branch.

---

## Merge Strategy Deep-Dive

When merging a PR on GitHub, three options exist:

| Strategy | What It Does | When to Use |
|----------|-------------|-------------|
| **Squash and merge** | Collapses all branch commits into **one** on main | **Default** — keeps main history clean |
| **Merge commit** | Preserves all individual commits + adds a merge commit | Multi-day features where each commit is a meaningful stage |
| **Rebase and merge** | Replays branch commits on top of main, no merge commit | Small PRs with already-clean commit history |

**Example — Squash merge:**
```
Your branch has:
  commit 1: "move files around"
  commit 2: "fix broken import"
  commit 3: "forgot to update config"

After squash merge, main gets ONE commit:
  "feat: add multi-sheet extraction support"

After regular merge, main gets 3 + 1 merge commit = 4 commits
```

**Why squash by default?**
- Most branch commits are incremental noise (`fix typo`, `forgot file`, `update again`)
- One commit per PR = clean `git log`, easy to `git revert`
- PR title becomes the commit message — it's human-readable at a glance

**When NOT to squash:**
- A long-running feature where individual commits tell a meaningful story (e.g., `step 1: add schema`, `step 2: add migration`, `step 3: add API endpoint`)

---

## Industry Branching Models

### GitHub Flow (Recommended for Most Teams)

Simplest model — `main` is always deployable:

```
main (always deployable)
  └── feature/branch-name → PR → main → deploy
```

This is the model described in SKILL.md. Best for teams deploying frequently.

### Git Flow (Enterprise Teams)

Adds a permanent `develop` integration branch:

```
main (production)
  └── release/* → main (after testing)
develop (integration)
  └── feature/* → develop
  └── hotfix/* → main + develop
```

Use this when you have formal QA cycles and infrequent releases.

### Trunk-Based Development (Google, Meta)

Ultra-simplified — everyone commits to `main` (trunk) via very short-lived branches (hours, not days):

```
main (trunk)
  └── short-lived branch (< 1 day) → PR → main
```

Key principles:
- Feature flags instead of long-lived branches
- Continuous deployment from trunk
- Branches exist for hours, not days

### GitLab Flow

Adds environment branches:

```
main → staging → production
  └── feature/* → main (auto-deploys to staging)
```

### Choosing the Right Prefix

```
Is it a new capability the user can see?
  → feat/

Is something broken?
  → fix/ (non-urgent) or hotfix/ (production is down)

Is the code changing but behavior stays the same?
  → refactor/

Is it dependencies, configs, or tooling?
  → chore/

Is it only documentation?
  → docs/
```

---

## Conventional Commits — Full Reference

### Branch Names vs Commit Message Types

The first 5 types (`feat`, `fix`, `refactor`, `chore`, `docs`) are used as both **branch prefixes** and **commit message types**. The remaining 5 (`perf`, `test`, `ci`, `build`, `revert`) are commit-message-only — work of that nature typically happens inside a `feat/` or `fix/` branch.

### With Ticket IDs

If your team uses a project tracker (Jira, Linear, GitHub Issues), include the ticket ID in branch names:

```
feat/PROJ-42-multi-sheet-extraction
fix/PROJ-67-api-null-response
chore/PROJ-80-upgrade-python-312
```

And in commit footers:
```
feat: add multi-sheet extraction support

Closes #42
Refs PROJ-42
```

### Breaking Changes

Signal breaking changes with `!` or a `BREAKING CHANGE` footer:

```
feat!: change extraction API response format

BREAKING CHANGE: The `results` key is now `extracted_rows`.
Update all consumers of the /extract endpoint.
```

---

## Common Scenario Recipes

### Scenario 1: Simple Bug Fix

```bash
git checkout main && git pull
git checkout -b fix/api-response-format
# ... make changes ...
git add . && git commit -m "fix: correct API response format for extraction results"
git push origin fix/api-response-format
# Open PR → main → squash and merge
# Deploy: git pull on server → rebuild → restart
```

### Scenario 2: New Feature

```bash
git checkout main && git pull
git checkout -b feat/new-file-type-support
# ... develop feature across multiple commits ...
git push origin feat/new-file-type-support
# Open PR → main → squash and merge
```

### Scenario 3: Preparing a Release

```bash
git checkout main && git pull
git checkout -b release/v0.34.0
git push origin release/v0.34.0
# Test thoroughly — only bug fix PRs into this branch now
git tag -a v0.34.0 -m "Release v0.34.0 — new file type support, API improvements"
git push origin v0.34.0
# Deploy from tag on server
# Merge release back into main
git checkout main && git merge release/v0.34.0 && git push
```

### Scenario 4: Production is Broken — Rollback

```bash
# On your server:
git fetch --all --tags
git tag -l                    # List available tags
git checkout v0.33.0          # Last working version
# Rebuild and restart
# Investigate and fix on a local hotfix branch
```

### Scenario 5: Compare What Changed Between Versions

```bash
# Diff between two tags
git diff v0.32.2..v0.33.0 --stat

# Commits between versions
git log v0.32.2..v0.33.0 --oneline

# What's on the server vs remote main
git log HEAD..origin/main --oneline

# What changed in a specific file between versions
git diff v0.32.2..v0.33.0 -- src/pipeline.py
```

### Scenario 6: Need to Work on Two Things Simultaneously

```bash
# Currently on feat/feature-a with uncommitted work
git stash                            # Shelve current changes
git checkout main && git pull
git checkout -b fix/urgent-fix       # Start urgent fix
# ... fix, commit, push, PR ...
git checkout feat/feature-a
git stash pop                        # Restore original work
```

---

## Deployment Recipes

### Standard Deployment (from main)

```bash
# SSH into server
cd ~/your-project

# Pull latest code
git checkout main
git pull origin main

# Rebuild and restart (Docker example)
docker compose build
docker compose up -d

# Verify
docker compose logs -f
```

### Deployment from a Specific Tag

```bash
git fetch --all --tags
git checkout v0.33.0
docker compose build
docker compose up -d
docker compose logs -f
```

### Pre-Deployment Checklist

- [ ] All PRs merged to `main` (or release branch)
- [ ] Build tested locally
- [ ] Release branch/tag created
- [ ] `.env` on server is up to date — check `.env.example` for new variables
- [ ] No uncommitted changes on server (`git status` is clean)
- [ ] Know which tag you're deploying and the previous tag (for rollback)

### Verify What's Currently Running

```bash
git log --oneline -1          # Current commit hash + message
git describe --tags           # Current tag (e.g., v0.33.0 or v0.33.0-3-gabcdef)
docker compose ps             # Running container status
```

### Docker Image Versioning (for faster rollbacks)

Tag Docker images alongside git tags for instant rollback without git checkout:

```bash
# Build and tag image with version
docker compose build
docker tag myapp:latest myapp:v0.33.0

# Rollback: swap the image tag
docker tag myapp:v0.32.2 myapp:latest
docker compose up -d          # No rebuild needed
```

---

## Branch Flow Diagram

```
feat/new-feature ──→ PR ──→ main ──→ release/v0.33.0 ──→ deploy
fix/some-bug ──────→ PR ──→ main ──→ (same release)   ──→ deploy
                                          │
                               hotfix ────→ fix on release branch
                                          │
                               tag v0.33.0 ←─── tested & approved
                                          │
                               merge back ──→ main
```

---

## Additional Resources

- [Conventional Commits Specification](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)
- [A Successful Git Branching Model (Git Flow)](https://nvie.com/posts/a-successful-git-branching-model/)
- [GitHub Flow Guide](https://docs.github.com/en/get-started/using-github/github-flow)
- [Trunk Based Development](https://trunkbaseddevelopment.com/)
- [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
- [Pro Git Book (free)](https://git-scm.com/book/en/v2)
