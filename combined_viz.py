"""
Combined Superorganism Visualizer
Loads both the global (17) and US (12) superorganism models and generates
a single interactive HTML with a toggle button to switch between views.

Toggle uses vis.js network.setData() — no page reload.
Served on localhost:8766 to avoid conflict with standalone global viz (8765).

Injection strategy:
  - Dataset script injected before </head> (after vis.js CDN load in head)
  - DataSet lines in drawGraph() replaced with pre-created references
  - UI + event JS injected before </body>
"""

import json
import os
import webbrowser
import http.server
import threading
from pyvis.network import Network

from superorganism_viz import (
    build_network,
    HEMISPHERE_COLORS, HEMISPHERE_BORDER,
    EXCITATORY_COLOR, INHIBITORY_COLOR, MIXED_COLOR,
    BG_COLOR, FONT_COLOR,
    LEGEND_HTML,
    TOOLTIP_CSS, TOOLTIP_DIV,
    edge_color_from_valences,
    serve_and_open,
)

# ---------------------------------------------------------------------------
# US-SPECIFIC CONSTANTS
# ---------------------------------------------------------------------------

# Border color override per weekly signal (concerning/notable nodes get colored rings)
SIGNAL_BORDER_COLOR = {
    "concerning": "#cf6679",   # red — matches inhibitory edge color
    "notable":    "#f5a623",   # amber
}

# Icon and color for the "This Week" section in node tooltips
SIGNAL_ICON = {
    "concerning": ("▼", "#cf6679"),
    "notable":    ("▲", "#f5a623"),
    "quiet":      ("—", "#8b949e"),
}


def us_node_size(rank: int) -> int:
    # rank 1 = 50, rank 12 = 19  (tighter range for 12 nodes)
    return int(50 - (rank - 1) * 2.8)


def load_latest_briefing(script_dir: str) -> dict | None:
    """Load the most recent weekly_briefing_*.json from briefings/, or None."""
    briefings_dir = os.path.join(script_dir, "briefings")
    if not os.path.isdir(briefings_dir):
        return None
    files = sorted(
        (f for f in os.listdir(briefings_dir)
         if f.startswith("weekly_briefing_") and f.endswith(".json")),
        reverse=True,
    )
    if not files:
        return None
    path = os.path.join(briefings_dir, files[0])
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_edge_signals_map(briefing: dict) -> dict:
    """
    Build {frozenset({name_a, name_b}): {ps_id: (valence, evidence)}}
    from briefing["edge_signals"].  Edge signals are news-sourced and override
    the default excitatory assumption for a given pair + DPS combination.
    """
    result = {}
    for sig in briefing.get("edge_signals", []):
        key = frozenset({sig["person_a"], sig["person_b"]})
        if key not in result:
            result[key] = {}
        result[key][sig["ps_id"]] = (sig["valence"], sig.get("evidence", ""))
    return result


# ---------------------------------------------------------------------------
# COMBINED-VIZ SPECIFIC HTML OVERLAYS
# ---------------------------------------------------------------------------

US_LEGEND_HTML = """
<div id="us-legend" style="
    display:none;
    position:fixed; top:20px; right:20px; z-index:9999;
    background:rgba(13,17,23,0.92); border:1px solid #30363d;
    border-radius:8px; padding:14px 18px; font-family:sans-serif;
    font-size:13px; color:#e6edf3; min-width:200px; line-height:1.8">
  <div style="font-size:14px;font-weight:700;margin-bottom:8px">US View</div>
  <div><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#4a90d9;margin-right:8px;vertical-align:middle"></span>All nodes: West (US)</div>
  <hr style="border-color:#30363d;margin:10px 0">
  <div style="font-size:14px;font-weight:700;margin-bottom:8px">Edges</div>
  <div><span style="display:inline-block;width:18px;height:3px;background:#4caf50;margin-right:8px;vertical-align:middle;border-radius:2px"></span>Excitatory</div>
  <div><span style="display:inline-block;width:18px;height:3px;background:#cf6679;margin-right:8px;vertical-align:middle;border-radius:2px"></span>Inhibitory</div>
  <div><span style="display:inline-block;width:18px;height:3px;background:#f5a623;margin-right:8px;vertical-align:middle;border-radius:2px"></span>Mixed</div>
  <hr style="border-color:#30363d;margin:10px 0">
  <div style="color:#8b949e;font-size:11px">
    Node size = influence rank<br>
    Edge width = # shared phase sequences<br>
    Hover nodes &amp; edges for detail
  </div>
</div>
"""

COMBINED_TITLE_HTML = """
<div id="graph-title-block" style="position:fixed;top:20px;left:20px;z-index:9999;font-family:sans-serif;color:#e6edf3">
  <div id="graph-title" style="font-size:20px;font-weight:bold;letter-spacing:0.5px">Human Superorganism</div>
  <div id="graph-subtitle" style="font-size:12px;color:#8b949e;margin-top:2px">17 Prime Movers &nbsp;&middot;&nbsp; Feb 2026</div>
</div>
"""

TOGGLE_BUTTON_HTML = """
<div id="view-toggle" style="
    position:fixed; top:20px; left:50%; transform:translateX(-50%);
    z-index:9999; font-family:sans-serif;">
  <button id="toggle-btn" onclick="switchView()"
      style="background:#238636; color:#fff; border:none; border-radius:6px;
             padding:8px 20px; font-size:13px; font-weight:600; cursor:pointer;
             letter-spacing:0.3px; box-shadow:0 2px 8px rgba(0,0,0,0.4)">
    Switch to US View (12)
  </button>
</div>
"""


# ---------------------------------------------------------------------------
# US NETWORK BUILDER
# ---------------------------------------------------------------------------

def compute_us_ps_valences(
    person_a: dict, person_b: dict, shared_ids: set,
    edge_signals_map: dict | None = None,
) -> list:
    """
    Returns list of (dps_id, name, valence) where valence is +1 or -1.
    Valence is news-driven: looks up edge_signals_map (from weekly briefing) first,
    defaults to +1 (excitatory) for any pair/DPS not covered by a news signal.
    """
    pair_key    = frozenset({person_a["name"], person_b["name"]})
    pair_signals = (edge_signals_map or {}).get(pair_key, {})

    ps_map = {ps["id"]: ps["name"]
              for ps in person_a["superorganism"].get("phase_sequences", [])}

    result = []
    for dps_id in sorted(shared_ids):
        name = ps_map.get(dps_id, dps_id)
        if dps_id in pair_signals:
            valence_str, _ = pair_signals[dps_id]
            v = -1 if valence_str == "adversarial" else +1
        else:
            v = +1
        result.append((dps_id, name, v))
    return result


def build_us_network(us_data: dict, briefing: dict | None = None):
    """
    Build the pyvis network for the US superorganism model.
    Returns (net, node_tooltips, edge_tooltips).
    All US nodes are West hemisphere (blue). Sectors replace regions in tooltips.
    If briefing is provided, applies news-driven edge valences and per-node
    weekly signal decorations (border color + "This Week" tooltip section).
    """
    net = Network(
        height="100vh",
        width="100%",
        bgcolor=BG_COLOR,
        font_color=FONT_COLOR,
        directed=False,
        notebook=False,
    )

    people        = us_data["superorganism_list"]
    node_tooltips = {}
    edge_tooltips = {}

    # All US actors are West
    color  = HEMISPHERE_COLORS["West"]
    border = HEMISPHERE_BORDER["West"]

    # Briefing-derived lookups
    edge_signals_map = build_edge_signals_map(briefing) if briefing else {}
    person_signals   = {}
    person_summaries = {}
    briefing_week    = ""
    if briefing:
        briefing_week = briefing.get("week_ending", "")
        for pu in briefing.get("person_updates", []):
            person_signals[pu["name"]]   = pu.get("signal", "quiet")
            person_summaries[pu["name"]] = pu.get("summary", "")

    # --- Nodes ---
    for person in people:
        so   = person["superorganism"]
        size = us_node_size(person["rank"])

        ps_rows = "".join(
            f"<div style='margin-top:6px'>"
            f"<span style='color:#58a6ff;font-weight:600;font-size:11px'>{ps['id']}</span>"
            f"&nbsp;<span style='color:#e6edf3'>{ps['name']}</span><br>"
            f"<span style='color:#8b949e;font-size:11px'>{ps['role']}</span>"
            f"</div>"
            for ps in so.get("phase_sequences", [])
        )
        asm_items = "".join(
            f"<div style='margin-top:3px;color:#8b949e;font-size:11px'>"
            f"&middot; {a['name']}"
            f"<span style='color:#6e7681'> — {a['role']}</span>"
            f"</div>"
            for a in so.get("cell_assemblies", [])
        )

        # Weekly signal decoration
        sig        = person_signals.get(person["name"], "quiet")
        node_border = SIGNAL_BORDER_COLOR.get(sig, border)
        sig_icon, sig_color = SIGNAL_ICON.get(sig, ("—", "#8b949e"))

        weekly_html = ""
        weekly_summary = person_summaries.get(person["name"], "")
        if weekly_summary:
            week_label = f"This Week ({briefing_week})" if briefing_week else "This Week"
            weekly_html = (
                f"<hr style='border-color:#30363d;margin:10px 0'>"
                f"<div style='font-size:12px;font-weight:600;color:{sig_color};"
                f"margin-bottom:4px'>{sig_icon} {week_label}</div>"
                f"<div style='color:#8b949e;font-size:11px'>{weekly_summary}</div>"
            )

        node_tooltips[person["name"]] = (
            f"<div style='max-width:360px;font-family:sans-serif;"
            f"font-size:13px;color:#e6edf3;line-height:1.5'>"
            f"<div style='font-size:16px;font-weight:700;margin-bottom:2px'>{person['name']}</div>"
            f"<div style='color:#8b949e;font-size:11px;margin-bottom:10px'>"
            f"{so.get('primary_sector','')} &nbsp;&middot;&nbsp; "
            f"<span style='color:{color}'>West</span></div>"
            f"<div style='color:#e6edf3;font-style:italic;margin-bottom:6px'>"
            f"{so.get('neuron_role','')}</div>"
            f"<div style='color:#8b949e;font-size:11px;margin-bottom:14px'>"
            f"{so.get('neuron_type','')}</div>"
            f"<div style='color:#e6edf3;font-weight:600;margin-bottom:2px'>Phase Sequences</div>"
            f"{ps_rows}"
            f"<div style='color:#e6edf3;font-weight:600;margin-top:14px;margin-bottom:2px'>"
            f"Cell Assemblies</div>"
            f"{asm_items}"
            f"{weekly_html}"
            f"</div>"
        )

        net.add_node(
            person["name"],
            label=person["name"],
            color={"background": color, "border": node_border,
                   "highlight": {"background": color, "border": "#ffffff"}},
            size=size,
            font={"size": 13, "color": FONT_COLOR, "face": "sans-serif"},
            borderWidth=3 if sig in ("concerning", "notable") else 2,
        )

    # --- Edges ---
    for i, a in enumerate(people):
        for b in people[i + 1:]:
            ps_a = {ps["id"] for ps in a["superorganism"].get("phase_sequences", [])}
            ps_b = {ps["id"] for ps in b["superorganism"].get("phase_sequences", [])}
            shared_ids = ps_a & ps_b
            if not shared_ids:
                continue

            valences    = compute_us_ps_valences(a, b, shared_ids, edge_signals_map)
            ecolor      = edge_color_from_valences(valences)
            count       = len(valences)
            excit_count = sum(1 for _, _, v in valences if v > 0)
            inhib_count = sum(1 for _, _, v in valences if v < 0)

            # Evidence text from the weekly briefing (if any)
            pair_key          = frozenset({a["name"], b["name"]})
            pair_signals_data = edge_signals_map.get(pair_key, {})

            header = f"Shared phase sequences ({count})"
            if inhib_count == 0:
                valence_label = "<span style='color:#4caf50'>All excitatory</span>"
            elif excit_count == 0:
                valence_label = "<span style='color:#cf6679'>All inhibitory</span>"
            else:
                valence_label = (
                    f"<span style='color:#4caf50'>{excit_count} excitatory</span>"
                    f" &nbsp;·&nbsp; "
                    f"<span style='color:#cf6679'>{inhib_count} inhibitory</span>"
                )

            ps_lines = ""
            for pid, name, v in valences:
                evidence = pair_signals_data.get(pid, (None, ""))[1]
                ev_html = (
                    f"<div style='color:#8b949e;font-size:10px;font-style:italic;"
                    f"padding-left:18px;margin-top:1px'>{evidence}</div>"
                    if evidence else ""
                )
                ps_lines += (
                    f"<div style='margin-top:5px'>"
                    f"<span style='font-size:14px;color:{'#4caf50' if v > 0 else '#cf6679'}'>"
                    f"{'⊕' if v > 0 else '⊖'}</span>"
                    f"&nbsp;<span style='color:#8b949e;font-size:11px'>{pid}</span>"
                    f"&nbsp;{name}"
                    f"{ev_html}"
                    f"</div>"
                )

            edge_key = f"{a['name']}|||{b['name']}"
            edge_tooltips[edge_key] = (
                f"<div style='font-family:sans-serif;font-size:13px;color:#e6edf3;"
                f"max-width:320px'>"
                f"<div style='font-weight:600;margin-bottom:4px'>{header}</div>"
                f"<div style='font-size:11px;margin-bottom:8px'>{valence_label}</div>"
                f"{ps_lines}"
                f"</div>"
            )

            net.add_edge(
                a["name"],
                b["name"],
                value=count,
                color={"color": ecolor, "highlight": "#ffffff", "opacity": 0.75},
            )

    net.set_options("""
    {
      "nodes": {
        "shape": "dot",
        "shadow": { "enabled": true, "size": 8 }
      },
      "edges": {
        "smooth": { "type": "continuous" },
        "shadow": false,
        "scaling": { "min": 1, "max": 8 }
      },
      "physics": {
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -60,
          "centralGravity": 0.005,
          "springLength": 200,
          "springConstant": 0.06,
          "damping": 0.9
        },
        "stabilization": { "iterations": 200 }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 9999999,
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """)

    return net, node_tooltips, edge_tooltips


# ---------------------------------------------------------------------------
# HTML EXTRACTION & INJECTION
# ---------------------------------------------------------------------------

def extract_dataset_json(html: str) -> tuple:
    """
    Extract JSON array content from the vis.DataSet lines.
    pyvis emits both on single (very long) lines.
    Returns (nodes_json_str, edges_json_str).
    """
    nodes_json = "[]"
    edges_json = "[]"

    for line in html.splitlines():
        stripped = line.strip()
        if stripped.startswith("nodes = new vis.DataSet("):
            start = stripped.find("[")
            end   = stripped.rfind("]") + 1
            if start >= 0 and end > start:
                nodes_json = stripped[start:end]
        elif stripped.startswith("edges = new vis.DataSet("):
            start = stripped.find("[")
            end   = stripped.rfind("]") + 1
            if start >= 0 and end > start:
                edges_json = stripped[start:end]

    return nodes_json, edges_json


def replace_dataset_lines(html: str) -> str:
    """
    Replace the two vis.DataSet lines inside drawGraph() with references
    to the pre-created global DataSet objects.
    Uses line-by-line scanning to avoid nested-bracket regex issues.
    """
    lines    = html.splitlines(keepends=True)
    new_lines = []
    for line in lines:
        stripped = line.strip()
        indent   = line[: len(line) - len(line.lstrip())]
        if stripped.startswith("nodes = new vis.DataSet("):
            new_lines.append(indent + "nodes = globalNodes;\n")
        elif stripped.startswith("edges = new vis.DataSet("):
            new_lines.append(indent + "edges = globalEdges;\n")
        else:
            new_lines.append(line)
    return "".join(new_lines)


def build_combined_html(
    global_net, global_node_tooltips, global_edge_tooltips,
    us_net,     us_node_tooltips,     us_edge_tooltips,
    output_path: str,
):
    """
    Write a single combined HTML with both networks and a toggle button.

    Injection order:
      </head> → TOOLTIP_CSS + dataset declarations script
      </body> → TOOLTIP_DIV + TOGGLE_BUTTON + US_LEGEND + GLOBAL_LEGEND +
                COMBINED_TITLE + tooltip dicts JS + tooltip listener JS + toggle JS
    """
    tmp_global = output_path + ".global.tmp.html"
    tmp_us     = output_path + ".us.tmp.html"

    try:
        global_net.write_html(tmp_global)
        us_net.write_html(tmp_us)

        with open(tmp_global, "r", encoding="utf-8") as f:
            html = f.read()
        with open(tmp_us, "r", encoding="utf-8") as f:
            us_html = f.read()
    finally:
        for p in (tmp_global, tmp_us):
            if os.path.exists(p):
                os.remove(p)

    global_nodes_json, global_edges_json = extract_dataset_json(html)
    us_nodes_json,     us_edges_json     = extract_dataset_json(us_html)

    # --- Step 1: Replace DataSet lines in drawGraph() ---
    html = replace_dataset_lines(html)

    # --- Step 2: Inject dataset script before </head> ---
    # vis.js CDN script is synchronous in <head>, so vis.DataSet is available here.
    dataset_script = f"""<script type="text/javascript">
// Combined superorganism datasets
var globalNodesData = {global_nodes_json};
var globalEdgesData = {global_edges_json};
var usNodesData     = {us_nodes_json};
var usEdgesData     = {us_edges_json};

// Pre-create DataSets for instant toggle (no re-parsing on switch)
var globalNodes = new vis.DataSet(globalNodesData);
var globalEdges = new vis.DataSet(globalEdgesData);
var usNodes     = new vis.DataSet(usNodesData);
var usEdges     = new vis.DataSet(usEdgesData);

var currentView = 'global';
</script>
"""
    html = html.replace("</head>", TOOLTIP_CSS + dataset_script + "</head>", 1)

    # --- Step 3: Serialize both tooltip dicts ---
    global_node_js = json.dumps(global_node_tooltips, ensure_ascii=False)
    global_edge_js = json.dumps(global_edge_tooltips, ensure_ascii=False)
    us_node_js     = json.dumps(us_node_tooltips, ensure_ascii=False)
    us_edge_js     = json.dumps(us_edge_tooltips, ensure_ascii=False)

    tooltip_dicts_js = f"""<script type="text/javascript">
var globalNodeTooltips = {global_node_js};
var globalEdgeTooltips = {global_edge_js};
var usNodeTooltips     = {us_node_js};
var usEdgeTooltips     = {us_edge_js};

var activeNodeTooltips = globalNodeTooltips;
var activeEdgeTooltips = globalEdgeTooltips;

window.updateTooltipMap = function(view) {{
  if (view === 'us') {{
    activeNodeTooltips = usNodeTooltips;
    activeEdgeTooltips = usEdgeTooltips;
  }} else {{
    activeNodeTooltips = globalNodeTooltips;
    activeEdgeTooltips = globalEdgeTooltips;
  }}
}};
window.getActiveNodeTooltips = function() {{ return activeNodeTooltips; }};
window.getActiveEdgeTooltips = function() {{ return activeEdgeTooltips; }};
</script>
"""

    # --- Step 4: Unified tooltip listener JS (polls for vis.js network object) ---
    tooltip_listener_js = """<script type="text/javascript">
(function() {
  var mouseX = 0, mouseY = 0;

  document.addEventListener('mousemove', function(e) {
    mouseX = e.clientX;
    mouseY = e.clientY;
    var tip = document.getElementById('custom-tooltip');
    if (tip && tip.style.display === 'block') positionTooltip(tip);
  });

  function positionTooltip(tip) {
    var pad = 18, tw = tip.offsetWidth, th = tip.offsetHeight;
    var vw = window.innerWidth, vh = window.innerHeight;
    var x = mouseX + pad, y = mouseY - pad;
    if (x + tw > vw - pad) x = mouseX - tw - pad;
    if (y + th > vh - pad) y = vh - th - pad;
    if (y < pad) y = pad;
    tip.style.left = x + 'px';
    tip.style.top  = y + 'px';
  }

  function showTooltip(html) {
    var tip = document.getElementById('custom-tooltip');
    tip.innerHTML = html;
    tip.style.display = 'block';
    positionTooltip(tip);
  }

  function hideTooltip() {
    var tip = document.getElementById('custom-tooltip');
    if (tip) tip.style.display = 'none';
  }

  var pollCount = 0;
  var poll = setInterval(function() {
    pollCount++;
    if (typeof network !== 'undefined' && network.body && network.on) {
      clearInterval(poll);
      attachListeners();
    }
    if (pollCount > 100) clearInterval(poll);
  }, 50);

  function attachListeners() {
    network.on('hoverNode', function(p) {
      var html = window.getActiveNodeTooltips()[p.node];
      if (html) showTooltip(html);
    });
    network.on('blurNode',  hideTooltip);

    network.on('hoverEdge', function(p) {
      var edge = network.body.data.edges.get(p.edge);
      if (!edge) return;
      var tips = window.getActiveEdgeTooltips();
      var html = tips[edge.from + '|||' + edge.to]
              || tips[edge.to   + '|||' + edge.from];
      if (html) showTooltip(html);
    });
    network.on('blurEdge',  hideTooltip);
    network.on('dragStart', hideTooltip);
  }
})();
</script>
"""

    # --- Step 5: Toggle JS ---
    toggle_js = """<script type="text/javascript">
window.switchView = function() {
  var btn         = document.getElementById('toggle-btn');
  var title       = document.getElementById('graph-title');
  var subtitle    = document.getElementById('graph-subtitle');
  var globalLegEl = document.getElementById('legend');
  var usLegEl     = document.getElementById('us-legend');

  if (currentView === 'global') {
    network.setData({nodes: usNodes, edges: usEdges});
    network.stabilize(150);
    currentView = 'us';
    window.updateTooltipMap('us');

    if (btn)         { btn.textContent = 'Switch to Global View (17)'; btn.style.background = '#6e40c9'; }
    if (title)       title.textContent    = 'US Superorganism';
    if (subtitle)    subtitle.textContent = '12 Domestic Prime Movers \u00b7 Feb 2026';
    if (globalLegEl) globalLegEl.style.display = 'none';
    if (usLegEl)     usLegEl.style.display     = 'block';

    // Refresh pyvis highlight-on-click state for new dataset
    allNodes = usNodes.get({returnType: 'Object'});
    allEdges = usEdges.get({returnType: 'Object'});
    nodeColors = {};
    for (var nid in allNodes) { nodeColors[nid] = allNodes[nid].color; }

  } else {
    network.setData({nodes: globalNodes, edges: globalEdges});
    network.stabilize(150);
    currentView = 'global';
    window.updateTooltipMap('global');

    if (btn)         { btn.textContent = 'Switch to US View (12)'; btn.style.background = '#238636'; }
    if (title)       title.textContent    = 'Human Superorganism';
    if (subtitle)    subtitle.textContent = '17 Prime Movers \u00b7 Feb 2026';
    if (globalLegEl) globalLegEl.style.display = 'block';
    if (usLegEl)     usLegEl.style.display     = 'none';

    // Refresh pyvis highlight-on-click state for restored dataset
    allNodes = globalNodes.get({returnType: 'Object'});
    allEdges = globalEdges.get({returnType: 'Object'});
    nodeColors = {};
    for (var nid in allNodes) { nodeColors[nid] = allNodes[nid].color; }
  }
};
</script>
"""

    # --- Step 6: Inject all UI + JS before </body> ---
    body_injection = (
        TOOLTIP_DIV
        + TOGGLE_BUTTON_HTML
        + US_LEGEND_HTML
        + LEGEND_HTML        # has id="legend" — global legend, visible by default
        + COMBINED_TITLE_HTML
        + tooltip_dicts_js
        + tooltip_listener_js
        + toggle_js
    )
    html = html.replace("</body>", body_injection + "</body>", 1)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    global_path  = os.path.join(script_dir, "superorganism_model.json")
    us_path      = os.path.join(script_dir, "us_superorganism_model.json")
    output_path  = os.path.join(script_dir, "combined_viz.html")

    print("=" * 60)
    print("COMBINED SUPERORGANISM VISUALIZER")
    print("=" * 60)

    # Guard: both model files must exist
    for path, label in [
        (global_path, "superorganism_model.json"),
        (us_path,     "us_superorganism_model.json"),
    ]:
        if not os.path.exists(path):
            print(f"\nERROR: {label} not found.")
            print("Run the corresponding pipeline first:")
            if "us_" in label:
                print("  python us_llm_council.py")
                print("  python us_superorganism_mapper.py")
            else:
                print("  python llm_council.py")
                print("  python superorganism_mapper.py")
            return

    with open(global_path, "r", encoding="utf-8") as f:
        global_data = json.load(f)
    with open(us_path, "r", encoding="utf-8") as f:
        us_data = json.load(f)

    print(f"\nGlobal model: {len(global_data['superorganism_list'])} nodes")
    print(f"US model:     {len(us_data['superorganism_list'])} nodes")

    print("\nBuilding global network...")
    global_net, global_node_tooltips, global_edge_tooltips = build_network(global_data)

    print("Loading latest briefing...")
    briefing = load_latest_briefing(script_dir)
    if briefing:
        n_signals = len(briefing.get("edge_signals", []))
        print(f"  Briefing: week ending {briefing.get('week_ending', 'unknown')} "
              f"({n_signals} edge signal{'s' if n_signals != 1 else ''})")
    else:
        print("  No briefing found — edges default to excitatory")

    print("Building US network...")
    us_net, us_node_tooltips, us_edge_tooltips = build_us_network(us_data, briefing=briefing)

    print("Assembling combined HTML...")
    build_combined_html(
        global_net, global_node_tooltips, global_edge_tooltips,
        us_net,     us_node_tooltips,     us_edge_tooltips,
        output_path,
    )
    print(f"Combined graph written to {output_path}")

    serve_and_open(output_path, port=8766)


if __name__ == "__main__":
    main()
