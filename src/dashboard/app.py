import glob
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
    margin-bottom: 16px;
    flex-wrap: wrap;
  }

  h1 { font-size: 20px; font-weight: 600; color: #ffffff; letter-spacing: 0.02em; }

  .meta { color: #888; font-size: 13px; }

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
    transition: background 0.15s;
  }
  .refresh-btn:hover { background: #1565c0; }

  /* ── Tab nav ─────────────────────────────────────────────────────────────── */
  .tab-nav {
    display: flex;
    gap: 0;
    margin-bottom: 20px;
    border-bottom: 1px solid #2a2a2a;
  }
  .tab-btn {
    padding: 9px 22px;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    color: #888;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.02em;
    transition: color 0.15s;
    margin-bottom: -1px;
  }
  .tab-btn:hover { color: #ccc; }
  .tab-btn.active { color: #fff; border-bottom-color: #1e88e5; }

  /* ── Legend ──────────────────────────────────────────────────────────────── */
  .legend {
    display: flex;
    gap: 20px;
    margin-bottom: 14px;
    font-size: 12px;
    color: #aaa;
  }
  .legend span { display: flex; align-items: center; gap: 6px; }
  .swatch { display: inline-block; width: 12px; height: 12px; border-radius: 2px; }
  .swatch-yellow { background: #856404; }
  .swatch-red    { background: #7f1d1d; }

  .no-data {
    color: #888;
    padding: 40px 0;
    text-align: center;
    font-size: 16px;
  }

  /* ── Tables ──────────────────────────────────────────────────────────────── */
  .table-wrap {
    overflow-x: auto;
    border-radius: 8px;
    border: 1px solid #2a2a2a;
    margin-bottom: 28px;
  }

  table { width: 100%; border-collapse: collapse; white-space: nowrap; }

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

  /* Predictions tab row states */
  tr.flag-row td          { background: #2a2000; }
  tr.flag-row:hover td    { background: #332800; }

  /* Evaluation tab row states */
  tr.outlier-row td       { background: #2d0a0a; }
  tr.outlier-row:hover td { background: #3a1010; }

  /* Flag / winner cells */
  td.flag-cell   { color: #ef5350; font-weight: 700; text-align: center; }
  td.win-model   { color: #66bb6a; font-weight: 600; }
  td.win-vegas   { color: #ef5350; font-weight: 600; }

  /* Diff colouring */
  td.diff-pos { color: #66bb6a; }
  td.diff-neg { color: #ef5350; }
  td.diff-neu { color: #aaa; }

  /* Numeric / player / team cells */
  td.num, th.num  { text-align: right; }
  td.team-cell    { font-size: 12px; font-weight: 600; color: #90caf9; letter-spacing: 0.04em; }
  td.player-cell  { font-weight: 500; color: #fff; }

  /* ── Evaluation summary cards ─────────────────────────────────────────────── */
  .cards {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 24px;
  }
  .card {
    background: #1e1e1e;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 14px 20px;
    min-width: 110px;
  }
  .card-label {
    font-size: 11px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .card-value {
    font-size: 22px;
    font-weight: 600;
    color: #fff;
    margin-top: 5px;
  }
  .card-sub {
    font-size: 11px;
    color: #666;
    margin-top: 2px;
  }

  /* ── Section titles ───────────────────────────────────────────────────────── */
  .section-title {
    font-size: 12px;
    font-weight: 600;
    color: #aaa;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin: 0 0 10px 0;
  }
</style>
</head>
<body>

<header>
  <h1>NBA Predictions</h1>
  <span class="meta">{{ date_label }}{% if player_count %} &mdash; {{ player_count }} players{% endif %}</span>
  <button class="refresh-btn" onclick="location.reload()">Refresh</button>
</header>

<nav class="tab-nav">
  <button class="tab-btn active" id="btn-predictions" onclick="showTab('predictions')">Today's Predictions</button>
  <button class="tab-btn"        id="btn-evaluation"  onclick="showTab('evaluation')">Last Evaluation</button>
</nav>

<!-- ══════════════════════════════════════════════════════════════════════════ -->
<!-- TAB: TODAY'S PREDICTIONS                                                   -->
<!-- ══════════════════════════════════════════════════════════════════════════ -->
<div id="tab-predictions">

{% if rows %}
<div class="legend">
  <span><span class="swatch swatch-yellow"></span> |PTS diff| &gt; 3.0 vs Vegas</span>
  <span><span class="swatch swatch-red"></span> Flag (*) = high-conviction disagreement</span>
</div>

<div class="table-wrap">
<table id="pred-table">
  <thead>
    <tr>
      <th onclick="sortTable('pred-table',0)">Player</th>
      <th onclick="sortTable('pred-table',1)">Team</th>
      <th onclick="sortTable('pred-table',2)">Matchup</th>
      <th class="num" onclick="sortTable('pred-table',3)">PTS</th>
      <th class="num" onclick="sortTable('pred-table',4)">PTS Range</th>
      <th class="num" onclick="sortTable('pred-table',5)">Vegas PTS</th>
      <th class="num" onclick="sortTable('pred-table',6)">PTS Diff</th>
      <th class="num" onclick="sortTable('pred-table',7)">REB</th>
      <th class="num" onclick="sortTable('pred-table',8)">REB Range</th>
      <th class="num" onclick="sortTable('pred-table',9)">Vegas REB</th>
      <th class="num" onclick="sortTable('pred-table',10)">REB Diff</th>
      <th class="num" onclick="sortTable('pred-table',11)">AST</th>
      <th class="num" onclick="sortTable('pred-table',12)">AST Range</th>
      <th class="num" onclick="sortTable('pred-table',13)">Vegas AST</th>
      <th class="num" onclick="sortTable('pred-table',14)">AST Diff</th>
      <th onclick="sortTable('pred-table',15)" style="text-align:center">Flag</th>
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
  {% if error %}{{ error }}{% else %}No prediction file found for today. Run predict.py first.{% endif %}
</div>
{% endif %}

</div><!-- /tab-predictions -->


<!-- ══════════════════════════════════════════════════════════════════════════ -->
<!-- TAB: LAST EVALUATION                                                        -->
<!-- ══════════════════════════════════════════════════════════════════════════ -->
<div id="tab-evaluation" style="display:none">

{% if eval %}

<!-- Section 1: Summary cards -->
<div class="cards">
  <div class="card">
    <div class="card-label">Date</div>
    <div class="card-value" style="font-size:16px;margin-top:7px">{{ eval.date_label }}</div>
  </div>
  <div class="card">
    <div class="card-label">Players</div>
    <div class="card-value">{{ eval.summary.n_eval }}</div>
    <div class="card-sub">evaluated</div>
  </div>
  <div class="card">
    <div class="card-label">PTS MAE</div>
    <div class="card-value">{{ eval.summary.pts_mae }}</div>
    <div class="card-sub">clean set</div>
  </div>
  <div class="card">
    <div class="card-label">REB MAE</div>
    <div class="card-value">{{ eval.summary.reb_mae }}</div>
    <div class="card-sub">clean set</div>
  </div>
  <div class="card">
    <div class="card-label">AST MAE</div>
    <div class="card-value">{{ eval.summary.ast_mae }}</div>
    <div class="card-sub">clean set</div>
  </div>
  <div class="card">
    <div class="card-label">PTS Bias</div>
    <div class="card-value">{{ eval.summary.pts_bias }}</div>
    <div class="card-sub">+ = over-predicted</div>
  </div>
  <div class="card">
    <div class="card-label">Model vs Vegas</div>
    <div class="card-value">{{ eval.summary.vs_vegas }}</div>
    <div class="card-sub">players won</div>
  </div>
</div>

<!-- Section 2: Model vs Vegas -->
{% if eval.vegas_rows %}
<p class="section-title">Model vs Vegas (players with both lines)</p>
<div class="table-wrap">
<table id="eval-vegas-table">
  <thead>
    <tr>
      <th onclick="sortTable('eval-vegas-table',0)">Player</th>
      <th class="num" onclick="sortTable('eval-vegas-table',1)">Predicted</th>
      <th class="num" onclick="sortTable('eval-vegas-table',2)">Vegas</th>
      <th class="num" onclick="sortTable('eval-vegas-table',3)">Actual</th>
      <th class="num" onclick="sortTable('eval-vegas-table',4)">Model Error</th>
      <th class="num" onclick="sortTable('eval-vegas-table',5)">Vegas Error</th>
      <th onclick="sortTable('eval-vegas-table',6)">Winner</th>
    </tr>
  </thead>
  <tbody>
  {% for r in eval.vegas_rows %}
    <tr>
      <td class="player-cell">{{ r.player }}</td>
      <td class="num">{{ r.pts_pred }}</td>
      <td class="num">{{ r.vegas_pts }}</td>
      <td class="num">{{ r.pts_actual }}</td>
      <td class="num">{{ r.model_err }}</td>
      <td class="num">{{ r.vegas_err }}</td>
      <td class="{{ r.winner_cls }}">{{ r.winner }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>
</div>
{% endif %}

<!-- Section 3: Best predictions -->
<p class="section-title">Top 10 Best Predictions (smallest PTS error)</p>
<div class="table-wrap">
<table id="eval-best-table">
  <thead>
    <tr>
      <th onclick="sortTable('eval-best-table',0)">Player</th>
      <th class="num" onclick="sortTable('eval-best-table',1)">Predicted</th>
      <th class="num" onclick="sortTable('eval-best-table',2)">Actual</th>
      <th class="num" onclick="sortTable('eval-best-table',3)">PTS Error</th>
    </tr>
  </thead>
  <tbody>
  {% for r in eval.best_rows %}
    <tr>
      <td class="player-cell">{{ r.player }}</td>
      <td class="num">{{ r.pts_pred }}</td>
      <td class="num">{{ r.pts_actual }}</td>
      <td class="num">{{ r.pts_error }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>
</div>

<!-- Section 4: Worst predictions -->
<p class="section-title">Top 10 Worst Predictions (largest PTS error) &mdash; red = likely DNP/injury</p>
<div class="table-wrap">
<table id="eval-worst-table">
  <thead>
    <tr>
      <th onclick="sortTable('eval-worst-table',0)">Player</th>
      <th class="num" onclick="sortTable('eval-worst-table',1)">Predicted</th>
      <th class="num" onclick="sortTable('eval-worst-table',2)">Actual</th>
      <th class="num" onclick="sortTable('eval-worst-table',3)">PTS Error</th>
    </tr>
  </thead>
  <tbody>
  {% for r in eval.worst_rows %}
    <tr class="{{ 'outlier-row' if r.is_outlier else '' }}">
      <td class="player-cell">{{ r.player }}</td>
      <td class="num">{{ r.pts_pred }}</td>
      <td class="num">{{ r.pts_actual }}</td>
      <td class="num">{{ r.pts_error }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>
</div>

{% else %}
<div class="no-data">
  {% if eval_error %}{{ eval_error }}{% else %}No evaluation file found. Run evaluate.py first.{% endif %}
</div>
{% endif %}

</div><!-- /tab-evaluation -->


<script>
// ── Tab switching ────────────────────────────────────────────────────────────
function showTab(name) {
  document.getElementById('tab-predictions').style.display = (name === 'predictions') ? '' : 'none';
  document.getElementById('tab-evaluation').style.display  = (name === 'evaluation')  ? '' : 'none';
  document.getElementById('btn-predictions').classList.toggle('active', name === 'predictions');
  document.getElementById('btn-evaluation').classList.toggle('active',  name === 'evaluation');
}

// ── Sortable tables ──────────────────────────────────────────────────────────
// Each table keeps its own sort state.
const sortState = {};

function sortTable(tableId, col) {
  const state = sortState[tableId] || { col: -1, dir: 1 };
  if (state.col === col) { state.dir = -state.dir; }
  else { state.col = col; state.dir = 1; }
  sortState[tableId] = state;

  const tbl   = document.getElementById(tableId);
  const ths   = tbl.querySelectorAll('th');
  const tbody = tbl.querySelector('tbody');
  const rows  = Array.from(tbody.querySelectorAll('tr'));

  ths.forEach((th, i) => {
    th.classList.remove('sorted-asc', 'sorted-desc');
    if (i === col) th.classList.add(state.dir === 1 ? 'sorted-asc' : 'sorted-desc');
  });

  rows.sort((a, b) => {
    const va = cellVal(a, col);
    const vb = cellVal(b, col);
    if (va < vb) return -state.dir;
    if (va > vb) return  state.dir;
    return 0;
  });

  rows.forEach(r => tbody.appendChild(r));
}

function cellVal(row, col) {
  const cell = row.querySelectorAll('td')[col];
  if (!cell) return '';
  const raw = cell.textContent.trim();
  // Strip range notation like "23.5 (13-34)" and +/* signs before parsing
  const leadNum = raw.replace(/[ ]*[(][^)]*[)]/, '').replace(/[+*]/g, '');
  const n = parseFloat(leadNum);
  return isNaN(n) ? raw.toLowerCase() : n;
}
</script>

</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
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


def _fmt_err(val) -> str:
    """Format an error value as '+X.X' / '-X.X', or '-' if missing."""
    try:
        if pd.isna(val):
            return "-"
        v = float(val)
        return f"+{v:.1f}" if v >= 0 else f"{v:.1f}"
    except Exception:
        return "-"


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_today_predictions():
    """Load today's prediction CSV. Returns (list_of_row_dicts, date_label, error_str)."""
    today = datetime.now().strftime("%Y-%m-%d")
    path  = os.path.join(PREDICTIONS_DIR, f"predictions_{today}.csv")

    if not os.path.exists(path):
        label = datetime.now().strftime("%B %d, %Y")
        return [], label, f"No predictions file found for {today}."

    df    = pd.read_csv(path)
    label = datetime.now().strftime("%B %d, %Y")
    rows  = []

    for _, r in df.iterrows():
        pts_diff_val = r.get("PTS_DIFF")
        try:
            is_flagged = not pd.isna(pts_diff_val) and abs(float(pts_diff_val)) > 3.0
        except Exception:
            is_flagged = False

        diff_str,     diff_cls     = _fmt_diff(pts_diff_val)
        reb_diff_str, reb_diff_cls = _fmt_diff(r.get("REB_DIFF"))
        ast_diff_str, ast_diff_cls = _fmt_diff(r.get("AST_DIFF"))

        matchup = str(r.get("MATCHUP", ""))

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

    rows.sort(key=lambda x: float(x["pts_pred"]) if x["pts_pred"] != "-" else -1,
              reverse=True)
    return rows, label, None


def load_latest_evaluation():
    """
    Find and load the most recent evaluation_{date}.csv.
    Returns (eval_data_dict, error_str). eval_data_dict is None on failure.
    """
    pattern = os.path.join(PREDICTIONS_DIR, "evaluation_*.csv")
    files   = sorted(glob.glob(pattern))  # lexicographic = chronological for YYYY-MM-DD
    if not files:
        return None, "No evaluation files found. Run evaluate.py first."

    latest   = files[-1]
    date_str = os.path.basename(latest).replace("evaluation_", "").replace(".csv", "")

    try:
        df = pd.read_csv(latest)
    except Exception as e:
        return None, f"Error loading {os.path.basename(latest)}: {e}"

    try:
        date_label = datetime.strptime(date_str, "%Y-%m-%d").strftime("%B %d, %Y")
    except Exception:
        date_label = date_str

    # ── Summary stats (clean set: outliers removed) ──────────────────────────
    matched  = df["PTS_ACTUAL"].notna()
    matched_df = df[matched].copy()
    matched_df["ABS_ERR"] = matched_df["PTS_ERROR"].abs()
    clean    = matched_df[matched_df["ABS_ERR"] <= 15.0]

    n_eval   = int(matched.sum())

    def _safe_mae(s):
        s = s.dropna()
        return f"{s.abs().mean():.2f}" if not s.empty else "-"

    def _safe_bias(s):
        s = s.dropna()
        if s.empty:
            return "-"
        v = s.mean()
        return f"+{v:.2f}" if v >= 0 else f"{v:.2f}"

    has_vegas = df["VEGAS_PTS"].notna() & matched
    n_vegas   = int(has_vegas.sum())
    n_beat    = int((df.loc[has_vegas, "MODEL_BEAT_VEGAS"].apply(
        lambda x: str(x).lower() == "true"
    )).sum())

    summary = {
        "n_eval":   n_eval,
        "pts_mae":  _safe_mae(clean["PTS_ERROR"])  if "PTS_ERROR" in clean.columns else "-",
        "reb_mae":  _safe_mae(clean["REB_ERROR"])  if "REB_ERROR" in clean.columns else "-",
        "ast_mae":  _safe_mae(clean["AST_ERROR"])  if "AST_ERROR" in clean.columns else "-",
        "pts_bias": _safe_bias(clean["PTS_ERROR"]) if "PTS_ERROR" in clean.columns else "-",
        "vs_vegas": f"{n_beat}/{n_vegas}" if n_vegas > 0 else "-",
    }

    # ── Section 2: model vs Vegas rows ───────────────────────────────────────
    vegas_rows = []
    vdf = matched_df[matched_df["VEGAS_PTS"].notna()].copy()
    vdf = vdf.sort_values("ABS_ERR", ascending=False)
    for _, r in vdf.iterrows():
        won = str(r.get("MODEL_BEAT_VEGAS", "")).lower() == "true"
        vegas_rows.append({
            "player":     str(r.get("PLAYER_NAME", "")),
            "pts_pred":   _fmt_pred(r.get("PTS_PRED")),
            "vegas_pts":  _fmt_pred(r.get("VEGAS_PTS")),
            "pts_actual": _fmt_pred(r.get("PTS_ACTUAL")),
            "model_err":  _fmt_err(r.get("PTS_ERROR")),
            "vegas_err":  _fmt_err(r.get("VEGAS_ERROR")),
            "winner":     "Model" if won else "Vegas",
            "winner_cls": "win-model" if won else "win-vegas",
        })

    # ── Section 3: best 10 ───────────────────────────────────────────────────
    best_rows = []
    for _, r in matched_df.nsmallest(10, "ABS_ERR").iterrows():
        best_rows.append({
            "player":     str(r.get("PLAYER_NAME", "")),
            "pts_pred":   _fmt_pred(r.get("PTS_PRED")),
            "pts_actual": _fmt_pred(r.get("PTS_ACTUAL")),
            "pts_error":  _fmt_err(r.get("PTS_ERROR")),
            "is_outlier": False,
        })

    # ── Section 4: worst 10 ──────────────────────────────────────────────────
    worst_rows = []
    for _, r in matched_df.nlargest(10, "ABS_ERR").iterrows():
        worst_rows.append({
            "player":     str(r.get("PLAYER_NAME", "")),
            "pts_pred":   _fmt_pred(r.get("PTS_PRED")),
            "pts_actual": _fmt_pred(r.get("PTS_ACTUAL")),
            "pts_error":  _fmt_err(r.get("PTS_ERROR")),
            "is_outlier": float(r["ABS_ERR"]) > 15.0,
        })

    eval_data = {
        "summary":    summary,
        "vegas_rows": vegas_rows,
        "best_rows":  best_rows,
        "worst_rows": worst_rows,
        "date_label": date_label,
    }
    return eval_data, None


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    rows, date_label, error = load_today_predictions()
    eval_data, eval_error   = load_latest_evaluation()
    return render_template_string(
        TEMPLATE,
        rows=rows,
        date_label=date_label,
        player_count=len(rows) if rows else None,
        error=error,
        eval=eval_data,
        eval_error=eval_error,
    )
