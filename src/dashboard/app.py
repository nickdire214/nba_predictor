import os
from datetime import datetime

import pandas as pd
from flask import Flask, render_template_string

app = Flask(__name__)

PREDICTIONS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "predictions"
)

# ─────────────────────────────────────────────────────────────────────────────
# HTML template
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NBA Predictions - {{ date_label }}</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #121212;
    color: #e0e0e0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 14px;
    padding: 24px;
  }

  header {
    display: flex;
    align-items: baseline;
    gap: 20px;
    margin-bottom: 20px;
    flex-wrap: wrap;
  }

  h1 {
    font-size: 20px;
    font-weight: 600;
    color: #ffffff;
    letter-spacing: 0.02em;
  }

  .meta {
    color: #888;
    font-size: 13px;
  }

  .refresh-btn {
    margin-left: auto;
    padding: 7px 18px;
    background: #1e88e5;
    color: #fff;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.02em;
    transition: background 0.15s;
  }
  .refresh-btn:hover { background: #1565c0; }

  .legend {
    display: flex;
    gap: 20px;
    margin-bottom: 14px;
    font-size: 12px;
    color: #aaa;
  }
  .legend span { display: flex; align-items: center; gap: 6px; }
  .swatch {
    display: inline-block;
    width: 12px; height: 12px;
    border-radius: 2px;
  }
  .swatch-yellow { background: #856404; }
  .swatch-red    { background: #7f1d1d; }

  .no-data {
    color: #888;
    padding: 40px 0;
    text-align: center;
    font-size: 16px;
  }

  .table-wrap {
    overflow-x: auto;
    border-radius: 8px;
    border: 1px solid #2a2a2a;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    white-space: nowrap;
  }

  thead tr { background: #1e1e1e; }

  th {
    padding: 10px 14px;
    text-align: left;
    font-size: 12px;
    font-weight: 600;
    color: #aaa;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    cursor: pointer;
    user-select: none;
    border-bottom: 1px solid #2a2a2a;
    white-space: nowrap;
  }
  th:hover { color: #fff; background: #252525; }
  th.sorted-asc::after  { content: " \u25b2"; font-size: 10px; color: #1e88e5; }
  th.sorted-desc::after { content: " \u25bc"; font-size: 10px; color: #1e88e5; }


  td {
    padding: 9px 14px;
    border-bottom: 1px solid #1e1e1e;
    color: #ddd;
  }

  tbody tr:hover td { background: #1a1a1a; }

  /* Big disagreement row highlight */
  tr.flag-row td { background: #2a2000; }
  tr.flag-row:hover td { background: #332800; }

  /* Flag cell */
  td.flag-cell {
    color: #ef5350;
    font-weight: 700;
    text-align: center;
  }

  /* Diff cell colouring */
  td.diff-pos { color: #66bb6a; }
  td.diff-neg { color: #ef5350; }
  td.diff-neu { color: #aaa; }

  /* Numeric columns right-aligned */
  td.num, th.num { text-align: right; }

  /* Team badge */
  td.team-cell {
    font-size: 12px;
    font-weight: 600;
    color: #90caf9;
    letter-spacing: 0.04em;
  }

  /* Player name bold */
  td.player-cell { font-weight: 500; color: #fff; }
</style>
</head>
<body>

<header>
  <h1>NBA Predictions</h1>
  <span class="meta">{{ date_label }}{% if player_count %} &mdash; {{ player_count }} players{% endif %}</span>
  <button class="refresh-btn" onclick="location.reload()">Refresh</button>
</header>

{% if rows %}
<div class="legend">
  <span><span class="swatch swatch-yellow"></span> |PTS diff| &gt; 3.0 vs Vegas</span>
  <span><span class="swatch swatch-red"></span> Flag (*) = high-conviction disagreement</span>
</div>

<div class="table-wrap">
<table id="pred-table">
  <thead>
    <tr>
      <th onclick="sortTable(0)">Player</th>
      <th onclick="sortTable(1)">Team</th>
      <th onclick="sortTable(2)">Matchup</th>
      <th class="num" onclick="sortTable(3)">PTS</th>
      <th class="num" onclick="sortTable(4)">PTS Range</th>
      <th class="num" onclick="sortTable(5)">Vegas PTS</th>
      <th class="num" onclick="sortTable(6)">PTS Diff</th>
      <th class="num" onclick="sortTable(7)">REB</th>
      <th class="num" onclick="sortTable(8)">REB Range</th>
      <th class="num" onclick="sortTable(9)">Vegas REB</th>
      <th class="num" onclick="sortTable(10)">REB Diff</th>
      <th class="num" onclick="sortTable(11)">AST</th>
      <th class="num" onclick="sortTable(12)">AST Range</th>
      <th class="num" onclick="sortTable(13)">Vegas AST</th>
      <th class="num" onclick="sortTable(14)">AST Diff</th>
      <th onclick="sortTable(15)" style="text-align:center">Flag</th>
    </tr>
  </thead>
  <tbody>
  {% for r in rows %}
    <tr class="{{ 'flag-row' if r.is_flagged else '' }}">
      <td class="player-cell">{{ r.player }}</td>
      <td class="team-cell">{{ r.team }}</td>
      <td>{{ r.matchup }}</td>
      <td class="num">{{ r.pts_pred }}</td>
      <td class="num">{{ r.pts_range }}</td>
      <td class="num">{{ r.vegas_pts }}</td>
      <td class="num {{ r.diff_class }}">{{ r.pts_diff }}</td>
      <td class="num">{{ r.reb_pred }}</td>
      <td class="num">{{ r.reb_range }}</td>
      <td class="num">{{ r.vegas_reb }}</td>
      <td class="num {{ r.reb_diff_class }}">{{ r.reb_diff }}</td>
      <td class="num">{{ r.ast_pred }}</td>
      <td class="num">{{ r.ast_range }}</td>
      <td class="num">{{ r.vegas_ast }}</td>
      <td class="num {{ r.ast_diff_class }}">{{ r.ast_diff }}</td>
      <td class="{{ 'flag-cell' if r.flag else '' }}" style="text-align:center">{{ r.flag }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>
</div>

{% else %}
<div class="no-data">
  {% if error %}
    {{ error }}
  {% else %}
    No prediction file found for today. Run predict.py first.
  {% endif %}
</div>
{% endif %}

<script>
// ── Sortable table ──────────────────────────────────────────────────────────
let sortCol = -1, sortDir = 1;

function sortTable(col) {
  const tbl  = document.getElementById('pred-table');
  const ths  = tbl.querySelectorAll('th');
  const tbody = tbl.querySelector('tbody');
  const rows  = Array.from(tbody.querySelectorAll('tr'));

  if (sortCol === col) {
    sortDir = -sortDir;
  } else {
    sortCol = col;
    sortDir = 1;
  }

  ths.forEach((th, i) => {
    th.classList.remove('sorted-asc', 'sorted-desc');
    if (i === col) th.classList.add(sortDir === 1 ? 'sorted-asc' : 'sorted-desc');
  });

  rows.sort((a, b) => {
    const va = cellVal(a, col);
    const vb = cellVal(b, col);
    if (va < vb) return -sortDir;
    if (va > vb) return  sortDir;
    return 0;
  });

  rows.forEach(r => tbody.appendChild(r));
}

function cellVal(row, col) {
  const cell = row.querySelectorAll('td')[col];
  if (!cell) return '';
  const raw = cell.textContent.trim();
  // Strip range notation like "23.5 (13-34)" -> keep 23.5 for sorting
  const leadNum = raw.replace(/[ ]*[(][^)]*[)]/, '').replace(/[+*]/g, '');
  const n = parseFloat(leadNum);
  return isNaN(n) ? raw.toLowerCase() : n;
}
</script>

</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_range(lo, hi) -> str:
    """Format p10-p90 range as '(13-34)', or '' if missing."""
    try:
        if pd.isna(lo) or pd.isna(hi):
            return ""
        return f"({int(round(lo))}-{int(round(hi))})"
    except Exception:
        return ""


def _fmt_pred(val) -> str:
    try:
        if pd.isna(val):
            return "-"
        return f"{float(val):.1f}"
    except Exception:
        return "-"


def _fmt_diff(val):
    """Returns (display_string, css_class)."""
    try:
        if pd.isna(val):
            return "-", "diff-neu"
        v = float(val)
        if v > 0:
            return f"+{v:.1f}", "diff-pos"
        elif v < 0:
            return f"{v:.1f}", "diff-neg"
        else:
            return "0.0", "diff-neu"
    except Exception:
        return "-", "diff-neu"


def load_today_predictions():
    """Load today's prediction CSV. Returns (list_of_row_dicts, date_label, error_str)."""
    today = datetime.now().strftime("%Y-%m-%d")
    path  = os.path.join(PREDICTIONS_DIR, f"predictions_{today}.csv")

    if not os.path.exists(path):
        label = datetime.now().strftime("%B %d, %Y")
        return [], label, f"No predictions file found for {today}."

    df = pd.read_csv(path)

    label = datetime.now().strftime("%B %d, %Y")
    rows  = []

    for _, r in df.iterrows():
        pts_diff_val = r.get("PTS_DIFF")
        try:
            is_flagged = not pd.isna(pts_diff_val) and abs(float(pts_diff_val)) > 3.0
        except Exception:
            is_flagged = False

        diff_str, diff_cls = _fmt_diff(pts_diff_val)
        reb_diff_str, reb_diff_cls = _fmt_diff(r.get("REB_DIFF"))
        ast_diff_str, ast_diff_cls = _fmt_diff(r.get("AST_DIFF"))

        # Trim matchup for display
        matchup = str(r.get("MATCHUP", ""))
        if len(matchup) > 14:
            matchup = matchup[-14:]

        rows.append({
            "player":         str(r.get("PLAYER_NAME", "")),
            "team":           str(r.get("TEAM_ABBREVIATION", "")),
            "matchup":        matchup,
            "pts_pred":       _fmt_pred(r.get("PTS_PRED")),
            "pts_range":      _fmt_range(r.get("PTS_LOW"), r.get("PTS_HIGH")),
            "vegas_pts":      _fmt_pred(r.get("VEGAS_PTS")),
            "pts_diff":       diff_str,
            "diff_class":     diff_cls,
            "reb_pred":       _fmt_pred(r.get("REB_PRED")),
            "reb_range":      _fmt_range(r.get("REB_LOW"), r.get("REB_HIGH")),
            "vegas_reb":      _fmt_pred(r.get("VEGAS_REB")),
            "reb_diff":       reb_diff_str,
            "reb_diff_class": reb_diff_cls,
            "ast_pred":       _fmt_pred(r.get("AST_PRED")),
            "ast_range":      _fmt_range(r.get("AST_LOW"), r.get("AST_HIGH")),
            "vegas_ast":      _fmt_pred(r.get("VEGAS_AST")),
            "ast_diff":       ast_diff_str,
            "ast_diff_class": ast_diff_cls,
            "flag":           "*" if is_flagged else "",
            "is_flagged":     is_flagged,
        })

    # Default sort: PTS_PRED descending
    rows.sort(key=lambda x: float(x["pts_pred"]) if x["pts_pred"] != "-" else -1,
              reverse=True)

    return rows, label, None


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    rows, date_label, error = load_today_predictions()
    return render_template_string(
        TEMPLATE,
        rows=rows,
        date_label=date_label,
        player_count=len(rows) if rows else None,
        error=error,
    )
