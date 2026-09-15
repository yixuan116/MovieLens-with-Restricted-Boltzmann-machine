#!/usr/bin/env python3
"""
Generate research_timesheet/index.html from real git history and real
Claude Code session activity. No hours are invented and no commit message
is reworded for effect -- every row is anchored to a real commit hash, and
weekly/monthly totals are whatever the real timestamps add up to.

How hours are estimated (documented so the number is checkable, not
asserted): for each repo, every git commit timestamp and every Claude
Code user/assistant message timestamp (matched to that repo via the
message's `cwd` field) is treated as one "activity point". Consecutive
points less than `session_gap_minutes` apart are merged into one working
session. A session's duration is (last point - first point) in that
session, floored at `min_block_minutes` so a single isolated commit still
gets a small, clearly-a-floor duration rather than 0. Only sessions that
contain at least one commit produce a timesheet row, so every row has a
citable hash in the Evidence column. Durations are rounded to the nearest
`round_to_minutes` for readability.

Usage:
    python3 generate_timesheet.py                # use config.json, write index.html, git commit
    python3 generate_timesheet.py --no-commit     # write index.html, skip the git commit
    python3 generate_timesheet.py --dry-run       # print summary, write nothing
    python3 generate_timesheet.py --config other.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, date as date_cls
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


# ---------------------------------------------------------------- config --

def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    cfg.setdefault("end_date", None)
    cfg.setdefault("claude_projects_dir", "~/.claude/projects")
    cfg.setdefault("session_gap_minutes", 90)
    cfg.setdefault("min_block_minutes", 20)
    cfg.setdefault("round_to_minutes", 15)
    cfg.setdefault("git_author_filter", None)
    return cfg


def resolve_repo_path(raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = (HERE / raw).resolve()
    return p


# ------------------------------------------------------------- git log ---

def get_git_commits(repo_path: Path, since: datetime, until: datetime, author_filter: str | None):
    """Return [{hash, dt (aware, local tz), subject}] for commits by the
    configured author (default: the repo's own git config user), newest
    first from git but returned oldest-first."""
    if not (repo_path / ".git").exists():
        print(f"  ! {repo_path} has no .git directory, skipping", file=sys.stderr)
        return []

    author = author_filter
    if author is None:
        try:
            author = subprocess.run(
                ["git", "-C", str(repo_path), "config", "user.email"],
                capture_output=True, text=True, check=True,
            ).stdout.strip() or None
        except subprocess.CalledProcessError:
            author = None

    cmd = [
        "git", "-C", str(repo_path), "log",
        f"--since={since.date().isoformat()}",
        f"--until={(until.date() + timedelta(days=1)).isoformat()}",
        "--date=iso-strict",
        "--pretty=format:%H%x1f%aI%x1f%s",
    ]
    if author:
        cmd.insert(4, f"--author={author}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ! git log failed in {repo_path}: {result.stderr.strip()}", file=sys.stderr)
        return []

    out = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        h, iso_dt, subject = line.split("\x1f", 2)
        dt = datetime.fromisoformat(iso_dt).astimezone()
        if dt < since or dt > until:
            continue
        out.append({"hash": h, "dt": dt, "subject": subject})
    out.sort(key=lambda c: c["dt"])
    return out


CONVENTIONAL_PREFIX = re.compile(r"^(feat|fix|docs|style|refactor|perf|test|chore|build|ci)(\([^)]*\))?:\s*", re.I)


def clean_subject(subject: str) -> str:
    """Strip a conventional-commit type prefix and capitalize the first
    letter. Does not add, remove, or reinterpret any claim in the message."""
    s = CONVENTIONAL_PREFIX.sub("", subject).strip()
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return s or subject


# --------------------------------------------------------- claude logs ---

def get_claude_activity(claude_projects_dir: Path, repo_path: Path, since: datetime, until: datetime):
    """Return a sorted list of aware local-tz datetimes for every
    user/assistant message whose `cwd` resolves to repo_path, across all
    session .jsonl files under claude_projects_dir."""
    if not claude_projects_dir.exists():
        return []

    repo_resolved = repo_path.resolve()
    timestamps = []

    for jsonl_path in claude_projects_dir.glob("*/*.jsonl"):
        try:
            with jsonl_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("type") not in ("user", "assistant"):
                        continue
                    cwd = rec.get("cwd")
                    if not cwd:
                        continue
                    try:
                        if Path(cwd).resolve() != repo_resolved:
                            continue
                    except OSError:
                        continue
                    ts = rec.get("timestamp")
                    if not ts:
                        continue
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone()
                    if since <= dt <= until:
                        timestamps.append(dt)
        except OSError:
            continue

    timestamps.sort()
    return timestamps


# --------------------------------------------------------- clustering ----

def round_minutes(minutes: float, to: int) -> float:
    return round(minutes / to) * to


def build_sessions(commits: list[dict], activity: list[datetime], gap_minutes: int,
                    min_minutes: int, round_minutes_to: int):
    """Merge commit points and bare activity points into sessions (gap <=
    gap_minutes stays in the same session). Return only sessions that
    contain >=1 commit, since every row needs citable evidence."""
    points = []
    for c in commits:
        points.append((c["dt"], c))
    for dt in activity:
        points.append((dt, None))
    points.sort(key=lambda p: p[0])

    sessions = []
    current = []
    for dt, commit in points:
        if current and (dt - current[-1][0]) > timedelta(minutes=gap_minutes):
            sessions.append(current)
            current = []
        current.append((dt, commit))
    if current:
        sessions.append(current)

    rows = []
    for sess in sessions:
        commits_in = [c for _, c in sess if c is not None]
        if not commits_in:
            continue
        start = sess[0][0]
        end = sess[-1][0]
        raw_minutes = max((end - start).total_seconds() / 60.0, 0)
        minutes = max(raw_minutes, min_minutes)
        minutes = max(round_minutes(minutes, round_minutes_to), round_minutes_to)
        hours = round(minutes / 60.0, 2)
        rows.append({
            "date": start.date(),
            "start": start,
            "end": end,
            "hours": hours,
            "commits": commits_in,
        })
    return rows


# ------------------------------------------------------------- html -----

WEEKLY_TARGET_NOTE = (
    "Hours are computed from real commit and session timestamps "
    "(see method note above); they are not normalized to any fixed weekly target."
)

CSS = """
:root {
  --ink: #16181c; --ink-soft: #4b4f57; --ink-faint: #7c8087;
  --border: #d8d4c8; --bg: #fbfaf7; --accent: #33506c; --stripe: #f2f0ea;
}
* { box-sizing: border-box; }
body {
  font-family: "Georgia", "Times New Roman", serif;
  background: var(--bg); color: var(--ink);
  max-width: 920px; margin: 0 auto; padding: 2.2rem 1.6rem 4rem;
  line-height: 1.45;
}
h1 { font-size: 1.4rem; margin: 0 0 0.2rem; }
.subtitle { color: var(--ink-soft); font-size: 0.92rem; margin: 0 0 1.2rem; }
.method {
  font-size: 0.78rem; color: var(--ink-faint); border: 1px solid var(--border);
  padding: 0.7rem 0.9rem; margin-bottom: 1.6rem; background: #fff;
}
.week-heading {
  font-size: 0.95rem; font-weight: bold; margin: 1.8rem 0 0.4rem;
  border-bottom: 2px solid var(--ink); padding-bottom: 0.2rem;
}
table { width: 100%; border-collapse: collapse; font-size: 0.82rem; margin-bottom: 0.3rem; }
th, td { border: 1px solid var(--border); padding: 0.35rem 0.5rem; text-align: left; vertical-align: top; }
th { background: var(--stripe); font-weight: bold; }
tr:nth-child(even) td { background: #fdfdfb; }
td.num, th.num { text-align: right; white-space: nowrap; }
tr.subtotal td { font-weight: bold; background: var(--stripe); }
.evidence { font-family: "Courier New", monospace; font-size: 0.74rem; color: var(--ink-soft); }
.signature-block {
  margin: 2rem 0 1rem; padding-top: 1rem; border-top: 1px dashed var(--border);
  display: flex; gap: 3rem; font-size: 0.85rem;
}
.sig-line { flex: 1; }
.sig-rule { border-bottom: 1px solid var(--ink); height: 1.6rem; margin-bottom: 0.2rem; }
.grand-total { margin-top: 2rem; font-size: 0.95rem; font-weight: bold; }
@media print {
  body { padding: 0; }
  .week-heading { break-before: auto; }
  table { break-inside: avoid; }
}
"""


def month_label(d: date_cls) -> str:
    return d.strftime("%B %Y")


def iso_week_start(d: date_cls) -> date_cls:
    return d - timedelta(days=d.weekday())


def render_html(cfg: dict, rows: list[dict]) -> str:
    rows = sorted(rows, key=lambda r: r["start"])

    weeks = defaultdict(list)
    for r in rows:
        weeks[iso_week_start(r["date"])].append(r)

    total_hours = round(sum(r["hours"] for r in rows), 2)
    start_label = cfg["start_date"]
    end_label = cfg.get("end_date") or datetime.now().date().isoformat()

    body_parts = []
    body_parts.append(f"<h1>Research Activity Timesheet</h1>")
    body_parts.append(
        f'<p class="subtitle">{escape(cfg["project_label"])} &middot; '
        f'{escape(start_label)} to {escape(end_label)} &middot; '
        f'generated {escape(datetime.now().strftime("%Y-%m-%d %H:%M"))}</p>'
    )
    body_parts.append(
        '<p class="method">Method: each row is a real working session reconstructed from git commit '
        "timestamps and Claude Code session activity in the matching repository's working directory. "
        "Consecutive activity within the configured session-gap window is merged into one session; a "
        "session's Hours is (last activity &minus; first activity) in that session, floored at the "
        "configured minimum block and rounded to the nearest quarter hour. Every row cites at least one "
        "git commit hash as evidence. " + WEEKLY_TARGET_NOTE + "</p>"
    )

    current_month = None
    week_starts_sorted = sorted(weeks.keys())
    for wk_start in week_starts_sorted:
        wk_rows = sorted(weeks[wk_start], key=lambda r: r["start"])
        wk_end = wk_start + timedelta(days=6)
        wk_hours = round(sum(r["hours"] for r in wk_rows), 2)

        this_month = (wk_start.year, wk_start.month)
        if current_month is not None and this_month != current_month:
            body_parts.append(render_signature_block(month_label(week_starts_sorted_month_anchor(current_month))))
        current_month = this_month

        body_parts.append(
            f'<div class="week-heading">Week of {wk_start.isoformat()} to {wk_end.isoformat()}</div>'
        )
        body_parts.append("<table>")
        body_parts.append(
            "<tr><th>Date</th><th>Time Block</th><th class=\"num\">Hours</th>"
            "<th>Project</th><th>Task Description</th><th>Evidence (commit)</th></tr>"
        )
        for r in wk_rows:
            date_str = r["start"].strftime("%a %Y-%m-%d")
            time_block = f'{r["start"].strftime("%H:%M")}&ndash;{r["end"].strftime("%H:%M")}'
            hashes = ", ".join(c["hash"][:7] for c in r["commits"])
            tasks = "; ".join(dict.fromkeys(clean_subject(c["subject"]) for c in r["commits"]))
            body_parts.append(
                "<tr>"
                f"<td>{escape(date_str)}</td>"
                f"<td>{time_block}</td>"
                f'<td class="num">{r["hours"]:.2f}</td>'
                f'<td>{escape(r["project"])}</td>'
                f"<td>{escape(tasks)}</td>"
                f'<td class="evidence">{escape(hashes)}</td>'
                "</tr>"
            )
        body_parts.append(
            f'<tr class="subtotal"><td colspan="2">Weekly subtotal</td>'
            f'<td class="num">{wk_hours:.2f}</td><td colspan="3"></td></tr>'
        )
        body_parts.append("</table>")

    if current_month is not None:
        body_parts.append(render_signature_block(month_label(week_starts_sorted_month_anchor(current_month))))

    body_parts.append(f'<p class="grand-total">Total hours, {escape(start_label)} to {escape(end_label)}: {total_hours:.2f}</p>')

    html = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>Research Timesheet</title><style>{CSS}</style></head><body>"
        + "".join(body_parts) +
        "</body></html>"
    )
    return html


def week_starts_sorted_month_anchor(month_tuple):
    """Helper: turn a (year, month) tuple back into a date for labeling."""
    return date_cls(month_tuple[0], month_tuple[1], 1)


def render_signature_block(month_lbl: str) -> str:
    return (
        f'<div class="signature-block">'
        f'<div class="sig-line"><div class="sig-rule"></div>Researcher signature &mdash; {escape(month_lbl)}</div>'
        f'<div class="sig-line"><div class="sig-rule"></div>Date</div>'
        f"</div>"
    )


# --------------------------------------------------------------- main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--no-commit", action="store_true", help="write index.html but skip git commit")
    ap.add_argument("--dry-run", action="store_true", help="print a summary, write nothing")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    since = datetime.fromisoformat(cfg["start_date"]).astimezone()
    until = (
        datetime.fromisoformat(cfg["end_date"]).astimezone()
        if cfg.get("end_date") else datetime.now().astimezone()
    )
    claude_dir = Path(cfg["claude_projects_dir"]).expanduser()

    all_rows = []
    for repo_cfg in cfg["repos"]:
        repo_path = resolve_repo_path(repo_cfg["path"])
        print(f"Scanning {repo_cfg['name']} ({repo_path}) ...")
        commits = get_git_commits(repo_path, since, until, cfg.get("git_author_filter"))
        activity = get_claude_activity(claude_dir, repo_path, since, until)
        print(f"  {len(commits)} commits, {len(activity)} Claude activity points in range")
        sessions = build_sessions(
            commits, activity,
            cfg["session_gap_minutes"], cfg["min_block_minutes"], cfg["round_to_minutes"],
        )
        for s in sessions:
            s["project"] = repo_cfg["name"]
        all_rows.extend(sessions)

    total_hours = round(sum(r["hours"] for r in all_rows), 2)
    print(f"\n{len(all_rows)} timesheet rows, {total_hours:.2f} total hours "
          f"({cfg['start_date']} to {cfg.get('end_date') or 'now'})")

    if args.dry_run:
        for r in sorted(all_rows, key=lambda r: r["start"]):
            print(f"  {r['start']:%Y-%m-%d %H:%M} - {r['end']:%H:%M}  {r['hours']:.2f}h  "
                  f"{r['project']}  [{', '.join(c['hash'][:7] for c in r['commits'])}]")
        return

    html = render_html(cfg, all_rows)
    out_path = HERE / "index.html"
    out_path.write_text(html)
    print(f"Wrote {out_path}")

    if not args.no_commit and all_rows:
        rel_path = out_path.relative_to(REPO_ROOT)
        msg = f"Update research timesheet [{cfg['start_date']} to {cfg.get('end_date') or datetime.now().date().isoformat()}]"
        subprocess.run(["git", "-C", str(REPO_ROOT), "add", str(rel_path)], check=True)
        diff = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "diff", "--cached", "--quiet"],
        )
        if diff.returncode == 0:
            print("No changes to commit.")
        else:
            subprocess.run(["git", "-C", str(REPO_ROOT), "commit", "-m", msg], check=True)
            print(f"Committed: {msg}")


if __name__ == "__main__":
    main()
