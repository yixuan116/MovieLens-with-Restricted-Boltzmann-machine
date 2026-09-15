# Research Timesheet

Generates `index.html`, a real activity log built from:

- Git commit timestamps in each configured repo (author-filtered to your own git config by default)
- Claude Code session activity timestamps (`~/.claude/projects/*/*.jsonl`) matched to a repo by the session's working directory

Every row in the output is anchored to at least one real commit hash. Hours are computed from the actual gap between the first and last activity timestamp in a session (see the "Method" note printed at the top of `index.html`), floored at a configurable minimum and rounded to the nearest quarter hour. Weekly and monthly totals are whatever those real numbers add up to -- the script does not normalize or pad any total, and commit messages are shown as written (only a conventional-commit prefix like `fix:` is stripped for readability).

## Configure

Edit `config.json`:

```json
{
  "start_date": "2026-09-05",
  "end_date": null,
  "project_label": "Computational Economics + Machine Learning Research",
  "repos": [
    { "path": "..", "name": "RBM MovieLens (Hyperbolic Number Systems)" },
    { "path": "/absolute/path/to/other-repo", "name": "Other Project" }
  ],
  "claude_projects_dir": "~/.claude/projects",
  "session_gap_minutes": 90,
  "min_block_minutes": 20,
  "round_to_minutes": 15,
  "git_author_filter": null
}
```

- `repos[].path` is resolved relative to this directory if not absolute.
- `git_author_filter` defaults to each repo's own `git config user.email`; set it explicitly if you commit under different identities in different repos.
- `end_date: null` means "through now" -- rerunning later just picks up whatever is new.

## Run

```bash
python3 generate_timesheet.py            # write index.html, then git commit it
python3 generate_timesheet.py --no-commit
python3 generate_timesheet.py --dry-run  # print the computed rows, write nothing
```

Rerunning is safe: the script always recomputes the full report from `start_date` to `end_date`/now, so it naturally picks up any new commits or session activity without any manual bookkeeping.
