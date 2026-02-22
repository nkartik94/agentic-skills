# Agentic Skills

Open collection of AI agent skills for coding agents — Claude Code, Codex, Cursor, Gemini CLI, and more.

Skills follow the [Agent Skills specification](https://github.com/anthropics/skills/blob/main/spec/agent-skills-spec.md) and work across any agent that supports the standard.

---

## Install

```bash
npx skills add nkartik94/agentic-skills
```

Or install a specific skill:

```bash
npx skills add nkartik94/agentic-skills --skill production-python
```

---

## Available Skills

| Skill | Description |
|-------|-------------|
| [production-python](skills/production-python/) | Production-grade Python coding conventions — module structure, section markers, type hints, Google docstrings, logging, naming, error handling, Pydantic, SQLAlchemy, CHANGELOG, README, and more |

---

## Contributing

PRs welcome! Each skill is a folder under `skills/` with a `SKILL.md` file.

### Adding a new skill

1. Create `skills/your-skill-name/SKILL.md`
2. Add YAML frontmatter with `name` and `description`
3. Write clear, actionable instructions in the markdown body
4. Open a PR

See the [Agent Skills spec](https://github.com/anthropics/skills/blob/main/spec/agent-skills-spec.md) for full format details.

---

## License

MIT
