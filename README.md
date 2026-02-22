# Agentic Skills

Open collection of AI agent skills for coding agents — Claude Code, Codex, Cursor, Gemini CLI, and more.

Skills are packaged instructions, conventions, and reference material that teach AI coding agents how to follow specific standards and workflows. They follow the [Agent Skills specification](https://agentskills.io/specification) and work across any agent that supports the standard.

## Install

```bash
npx skills add nkartik94/agentic-skills
```

Or install a specific skill:

```bash
npx skills add nkartik94/agentic-skills --skill production-python
```

Skills activate automatically when the agent detects a relevant task — no manual triggering needed.

## Available Skills

| Skill | Description | Highlights |
|-------|-------------|------------|
| [production-python](skills/production-python/) | Production-grade Python coding conventions | 21 rules covering module structure, type hints, docstrings, logging, formatting, error handling, architecture patterns, testing, Pydantic, SQLAlchemy, and more |

## How Skills Work

1. **Discovery** — The agent reads each skill's `name` and `description` from the YAML frontmatter
2. **Activation** — When a task matches a skill's description, the agent loads the full `SKILL.md` instructions
3. **Progressive loading** — Detailed references (`references/`, `scripts/`, `assets/`) are loaded only when needed, keeping context efficient

## Compatibility

Skills work with any AI coding agent that supports the [Agent Skills format](https://agentskills.io/specification):

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code)
- [Cursor](https://www.cursor.com/)
- [Codex](https://openai.com/index/codex/)
- [Gemini CLI](https://github.com/google-gemini/gemini-cli)
- [Cline](https://cline.bot/)
- And any other agent implementing the standard

## Skill Structure

Each skill is a self-contained folder under `skills/`:

```
skills/
└── your-skill-name/
    ├── SKILL.md              # Required — frontmatter + instructions
    ├── references/           # Optional — detailed docs loaded on demand
    │   └── REFERENCE.md
    ├── scripts/              # Optional — executable helpers
    └── assets/               # Optional — templates, schemas, data files
```

## Contributing

PRs welcome! To add a new skill:

1. Create `skills/your-skill-name/SKILL.md`
2. Add YAML frontmatter:
   ```yaml
   ---
   name: your-skill-name
   description: What the skill does and when to use it. Include trigger phrases.
   license: MIT
   metadata:
     author: your-name
     version: "1.0.0"
   ---
   ```
3. Write clear, actionable instructions in the markdown body
4. Keep `SKILL.md` under 500 lines — move detailed content to `references/`
5. Open a PR

### Guidelines

- **`name`** must be lowercase, hyphens only, and match the folder name
- **`description`** should explain both *what* the skill does and *when* to activate it (max 1024 chars)
- Include a `## When to Apply` section at the top of the body
- Use `references/REFERENCE.md` for templates, full implementations, and extended examples
- Test your skill by installing it locally and verifying activation on relevant tasks

See the [Agent Skills specification](https://agentskills.io/specification) for the full format reference.

## Links

- [Agent Skills Specification](https://agentskills.io/specification)
- [skills.sh — Skill Directory](https://skills.sh)
- [Anthropic Skills Repository](https://github.com/anthropics/skills)
- [Vercel Agent Skills](https://github.com/vercel-labs/agent-skills)

## License

MIT
