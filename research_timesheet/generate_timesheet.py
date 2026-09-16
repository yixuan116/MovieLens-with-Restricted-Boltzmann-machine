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

def extract_preview(rec: dict) -> str | None:
    """Pull real, unmodified text out of a user-message record (first text
    block), for use as a factual topic label. Returns None for anything
    that isn't plain text (tool results, images, etc.)."""
    msg = rec.get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = None
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                break
        if text is None:
            return None
    else:
        return None
    text = " ".join(text.split())
    if not text or text.startswith("<"):
        # System-injected notes (<ide_opened_file>, <system-reminder>, etc.),
        # not something the person actually typed -- not a real topic label.
        return None
    return text[:140]


def get_claude_activity(claude_projects_dir: Path, repo_path: Path, since: datetime, until: datetime):
    """Return a sorted list of {dt, preview} for every user/assistant
    message whose `cwd` resolves to repo_path, across all session .jsonl
    files under claude_projects_dir. `preview` is the real message text
    (truncated) for user messages, None for assistant messages -- it is
    never generated or reworded, only lifted verbatim from the log."""
    if not claude_projects_dir.exists():
        return []

    repo_resolved = repo_path.resolve()
    points = []

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
                        cwd_resolved = Path(cwd).resolve()
                    except OSError:
                        continue
                    # A repo can get reorganized into a subdirectory mid-project;
                    # sessions logged against the old parent cwd still belong to it.
                    if cwd_resolved != repo_resolved and cwd_resolved != repo_resolved.parent:
                        continue
                    ts = rec.get("timestamp")
                    if not ts:
                        continue
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone()
                    if since <= dt <= until:
                        preview = extract_preview(rec) if rec.get("type") == "user" else None
                        points.append({"dt": dt, "preview": preview})
        except OSError:
            continue

    points.sort(key=lambda p: p["dt"])
    return points


# --------------------------------------------------------- clustering ----

def round_minutes(minutes: float, to: int) -> float:
    return round(minutes / to) * to


def build_sessions(commits: list[dict], activity: list[dict], gap_minutes: int,
                    min_minutes: int, round_minutes_to: int):
    """Merge commit points and Claude activity points into sessions (gap <=
    gap_minutes stays in the same session). Every session that has real
    activity in it becomes a row: sessions with >=1 commit are evidenced by
    the commit hash(es); sessions with only Claude activity and no commit
    are still real, timestamped work -- evidenced by the session's own
    logged time range and a topic line lifted verbatim from the first real
    user message in that window (never generated or reworded)."""
    points = []
    for c in commits:
        points.append((c["dt"], c, None))
    for a in activity:
        points.append((a["dt"], None, a.get("preview")))
    points.sort(key=lambda p: p[0])

    sessions = []
    current = []
    for dt, commit, preview in points:
        if current and (dt - current[-1][0]) > timedelta(minutes=gap_minutes):
            sessions.append(current)
            current = []
        current.append((dt, commit, preview))
    if current:
        sessions.append(current)

    rows = []
    for sess in sessions:
        commits_in = [c for _, c, _ in sess if c is not None]
        start = sess[0][0]
        end = sess[-1][0]
        raw_minutes = max((end - start).total_seconds() / 60.0, 0)
        minutes = max(raw_minutes, min_minutes)
        minutes = max(round_minutes(minutes, round_minutes_to), round_minutes_to)
        hours = round(minutes / 60.0, 2)

        if commits_in:
            evidence_kind = "commit"
        else:
            evidence_kind = "chatlog"

        first_preview = next((p for _, _, p in sess if p), None)

        rows.append({
            "date": start.date(),
            "start": start,
            "end": end,
            "hours": hours,
            "commits": commits_in,
            "evidence_kind": evidence_kind,
            "topic_preview": first_preview,
        })
    return rows


def load_manual_entries(path: Path, default_project: str, since: datetime, until: datetime):
    """Read research_timesheet/manual_entries.json (a plain, version-controlled
    file people can hand-edit or paste the page's "Export as JSON" output
    into) and turn each entry into a row in the same shape build_sessions
    produces, so it's a permanent, merged part of the report and its total
    -- not a browser-only draft."""
    if not path.exists():
        return []
    try:
        entries = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"  ! could not parse {path}: {e}", file=sys.stderr)
        return []

    rows = []
    for e in entries:
        try:
            d = date_cls.fromisoformat(e["date"])
            sh, sm = (int(x) for x in e["start"].split(":"))
            eh, em = (int(x) for x in e["end"].split(":"))
        except (KeyError, ValueError) as err:
            print(f"  ! skipping malformed manual entry {e!r}: {err}", file=sys.stderr)
            continue
        start = datetime(d.year, d.month, d.day, sh, sm).astimezone()
        end = datetime(d.year, d.month, d.day, eh, em).astimezone()
        if end < start:
            end += timedelta(days=1)
        if not (since <= start <= until):
            continue
        hours = round(max((end - start).total_seconds() / 3600.0, 0), 2)
        rows.append({
            "date": start.date(),
            "start": start,
            "end": end,
            "hours": hours,
            "commits": [],
            "evidence_kind": "manual",
            "topic_preview": e.get("task", ""),
            "project": e.get("project") or default_project,
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
.group-label {
  font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.04em;
  color: var(--ink-faint); margin: 0.9rem 0 0.25rem;
}
.week-total { text-align: right; font-size: 0.85rem; font-weight: bold; margin: 0.3rem 0 0; }
.grand-total { margin-top: 2rem; font-size: 0.95rem; font-weight: bold; }
.manual-section {
  margin-top: 2.4rem; padding-top: 1.2rem; border-top: 2px solid var(--ink);
}
.manual-form {
  display: grid; grid-template-columns: 8rem 5.5rem 5.5rem 10rem 1fr 6rem;
  gap: 0.4rem; margin-bottom: 0.6rem; font-size: 0.8rem;
}
.manual-form input, .manual-form button {
  font-family: inherit; font-size: 0.8rem; padding: 0.3rem 0.4rem;
  border: 1px solid var(--border); background: #fff;
}
.manual-form button {
  background: var(--accent); color: #fff; border-color: var(--accent); cursor: pointer;
}
.manual-note { font-size: 0.74rem; color: var(--ink-faint); margin: 0 0 0.8rem; }
.manual-actions { margin-top: 0.5rem; display: flex; gap: 0.6rem; align-items: center; }
.manual-actions button {
  font-family: inherit; font-size: 0.76rem; padding: 0.3rem 0.7rem;
  border: 1px solid var(--border); background: #fff; cursor: pointer;
}
.manual-row-del { color: #a33; cursor: pointer; font-size: 0.76rem; background: none; border: none; text-decoration: underline; }
#manual-export { width: 100%; font-family: "Courier New", monospace; font-size: 0.7rem; margin-top: 0.5rem; height: 4rem; display: none; }
@media print {
  .manual-form, .manual-actions, .manual-row-del { display: none !important; }
}
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
        "configured minimum block and rounded to the nearest quarter hour. Sessions that produced a commit "
        "reference that commit's hash; sessions that did not reference the session's own logged time range, "
        "with the Task Description lifted verbatim from the first real message in that window "
        "(never generated or reworded). " + WEEKLY_TARGET_NOTE + "</p>"
    )

    body_parts.append(MANUAL_SECTION_HTML)
    body_parts.append(MANUAL_SECTION_JS)

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

        commit_rows = [r for r in wk_rows if r["evidence_kind"] == "commit"]
        chatlog_rows = [r for r in wk_rows if r["evidence_kind"] == "chatlog"]
        manual_rows_wk = [r for r in wk_rows if r["evidence_kind"] == "manual"]

        for group_label, group_rows in (
            ("With commit", commit_rows),
            ("Without commit (chat log only)", chatlog_rows),
            ("Manually logged", manual_rows_wk),
        ):
            if not group_rows:
                continue
            group_hours = round(sum(r["hours"] for r in group_rows), 2)
            body_parts.append(f'<p class="group-label">{escape(group_label)}</p>')
            body_parts.append("<table>")
            body_parts.append(
                "<tr><th>Date</th><th>Time Block</th><th class=\"num\">Hours</th>"
                "<th>Project</th><th>Task Description</th><th>Reference</th></tr>"
            )
            for r in group_rows:
                date_str = r["start"].strftime("%a %Y-%m-%d")
                time_block = f'{r["start"].strftime("%H:%M")}&ndash;{r["end"].strftime("%H:%M")}'
                if r["evidence_kind"] == "commit":
                    evidence = ", ".join(c["hash"][:7] for c in r["commits"])
                    tasks = "; ".join(dict.fromkeys(clean_subject(c["subject"]) for c in r["commits"]))
                elif r["evidence_kind"] == "manual":
                    evidence = "manual_entries.json"
                    tasks = r.get("topic_preview") or ""
                else:
                    evidence = f'chat log {r["start"].strftime("%H:%M")}–{r["end"].strftime("%H:%M")}'
                    tasks = r.get("topic_preview") or "(session logged; no message text captured)"
                body_parts.append(
                    "<tr>"
                    f"<td>{escape(date_str)}</td>"
                    f"<td>{time_block}</td>"
                    f'<td class="num">{r["hours"]:.2f}</td>'
                    f'<td>{escape(r["project"])}</td>'
                    f"<td>{escape(tasks)}</td>"
                    f'<td class="evidence">{escape(evidence)}</td>'
                    "</tr>"
                )
            body_parts.append(
                f'<tr class="subtotal"><td colspan="2">{escape(group_label)} subtotal</td>'
                f'<td class="num">{group_hours:.2f}</td><td colspan="3"></td></tr>'
            )
            body_parts.append("</table>")

        body_parts.append(f'<p class="week-total">Week total: {wk_hours:.2f}h</p>')

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


MANUAL_SECTION_HTML = """
<div class="manual-section">
  <div class="week-heading">Add an entry</div>
  <p class="manual-note">Type an entry below, then paste the Export output into
  <code>research_timesheet/manual_entries.json</code> and rerun the script -- it becomes a permanent
  "Manually logged" row in the week tables above and is included in the total, the same as any other row.
  Until then it's just a draft saved in this browser.</p>
  <div class="manual-form">
    <input type="date" id="m-date">
    <input type="time" id="m-start" placeholder="start">
    <input type="time" id="m-end" placeholder="end">
    <input type="text" id="m-project" placeholder="Project">
    <input type="text" id="m-task" placeholder="Task description">
    <button type="button" id="m-add">Add entry</button>
  </div>
  <table>
    <tr><th>Date</th><th>Time Block</th><th class="num">Hours</th><th>Project</th><th>Task Description</th><th></th></tr>
    <tbody id="manual-tbody"></tbody>
    <tr class="subtotal"><td colspan="2">Draft subtotal (not yet in manual_entries.json)</td>
      <td class="num" id="manual-subtotal">0.00</td><td colspan="3"></td></tr>
  </table>
  <div class="manual-actions">
    <button type="button" id="m-export">Export as JSON</button>
    <button type="button" id="m-clear">Clear drafts</button>
  </div>
  <textarea id="manual-export" readonly></textarea>
</div>
"""

MANUAL_SECTION_JS = """
<script>
(function () {
  var KEY = "research_timesheet_manual_entries_v1";

  function load() {
    try { return JSON.parse(localStorage.getItem(KEY) || "[]"); }
    catch (e) { return []; }
  }
  function save(entries) {
    try { localStorage.setItem(KEY, JSON.stringify(entries)); } catch (e) {}
  }
  function hoursBetween(start, end) {
    var s = start.split(":").map(Number), e = end.split(":").map(Number);
    var mins = (e[0] * 60 + e[1]) - (s[0] * 60 + s[1]);
    if (mins < 0) mins += 24 * 60;
    return Math.round((mins / 60) * 100) / 100;
  }

  function render() {
    var entries = load();
    var tbody = document.getElementById("manual-tbody");
    tbody.innerHTML = "";
    var subtotal = 0;
    entries.forEach(function (e, i) {
      subtotal += e.hours;
      var tr = document.createElement("tr");
      tr.innerHTML =
        "<td>" + e.date + "</td>" +
        "<td>" + e.start + "\\u2013" + e.end + "</td>" +
        "<td class=\\"num\\">" + e.hours.toFixed(2) + "</td>" +
        "<td></td><td></td><td></td>";
      tr.children[3].textContent = e.project;
      tr.children[4].textContent = e.task;
      var delBtn = document.createElement("button");
      delBtn.className = "manual-row-del";
      delBtn.textContent = "remove";
      delBtn.addEventListener("click", function () {
        var cur = load();
        cur.splice(i, 1);
        save(cur);
        render();
      });
      tr.children[5].appendChild(delBtn);
      tbody.appendChild(tr);
    });
    document.getElementById("manual-subtotal").textContent = subtotal.toFixed(2);
  }

  document.getElementById("m-add").addEventListener("click", function () {
    var date = document.getElementById("m-date").value;
    var start = document.getElementById("m-start").value;
    var end = document.getElementById("m-end").value;
    var project = document.getElementById("m-project").value.trim();
    var task = document.getElementById("m-task").value.trim();
    if (!date || !start || !end || !task) {
      alert("Date, start, end, and task description are required.");
      return;
    }
    var entries = load();
    entries.push({ date: date, start: start, end: end, hours: hoursBetween(start, end), project: project, task: task });
    save(entries);
    document.getElementById("m-task").value = "";
    render();
  });

  document.getElementById("m-clear").addEventListener("click", function () {
    if (!confirm("Remove all additional entries in this browser?")) return;
    save([]);
    render();
  });

  document.getElementById("m-export").addEventListener("click", function () {
    var box = document.getElementById("manual-export");
    box.value = JSON.stringify(load(), null, 2);
    box.style.display = box.style.display === "none" ? "block" : "none";
  });

  render();
})();
</script>
"""


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

    manual_path = HERE / "manual_entries.json"
    manual_rows = load_manual_entries(manual_path, cfg["repos"][0]["name"] if cfg["repos"] else "", since, until)
    if manual_rows:
        print(f"  {len(manual_rows)} manually-logged entries from {manual_path.name}")
    all_rows.extend(manual_rows)

    total_hours = round(sum(r["hours"] for r in all_rows), 2)
    print(f"\n{len(all_rows)} timesheet rows, {total_hours:.2f} total hours "
          f"({cfg['start_date']} to {cfg.get('end_date') or 'now'})")

    if args.dry_run:
        for r in sorted(all_rows, key=lambda r: r["start"]):
            if r["commits"]:
                ref = ", ".join(c["hash"][:7] for c in r["commits"])
            elif r["evidence_kind"] == "manual":
                ref = "manual entry"
            else:
                ref = f'chatlog {r["start"]:%H:%M}-{r["end"]:%H:%M}'
            print(f"  {r['start']:%Y-%m-%d %H:%M} - {r['end']:%H:%M}  {r['hours']:.2f}h  "
                  f"{r['project']}  [{ref}]")
        return

    html = render_html(cfg, all_rows)
    out_path = REPO_ROOT / cfg.get("output_path", "docs/research_timesheet.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
