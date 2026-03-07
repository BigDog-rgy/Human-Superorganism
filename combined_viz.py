"""
Combined Superorganism Visualizer

Loads the US model and optionally the global model, then generates a single
interactive HTML with a toggle button to switch between views.

Toggle uses vis.js network.setData() — no page reload.
Served on localhost:8766.

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


# ---------------------------------------------------------------------------
# SHARED VISUAL CONSTANTS
# ---------------------------------------------------------------------------

HEMISPHERE_COLORS = {
    "West":   "#4a90d9",   # blue
    "East":   "#e05c5c",   # red
    "Bridge": "#a78bfa",   # purple
    "Global": "#8b949e",   # grey
}

HEMISPHERE_BORDER = {
    "West":   "#2d6da8",
    "East":   "#b03030",
    "Bridge": "#7c5cbf",
    "Global": "#6e7681",
}

EXCITATORY_COLOR = "#4caf50"   # green
INHIBITORY_COLOR = "#cf6679"   # red
MIXED_COLOR      = "#f5a623"   # amber

BG_COLOR   = "#0d1117"
FONT_COLOR = "#e6edf3"

# ---------------------------------------------------------------------------
# TOOLTIP HTML / CSS
# ---------------------------------------------------------------------------

TOOLTIP_CSS = """<style>
#custom-tooltip {
  display: none;
  position: fixed;
  z-index: 99999;
  background: rgba(13, 17, 23, 0.97);
  border: 1px solid #30363d;
  border-radius: 8px;
  padding: 12px 16px;
  pointer-events: none;
  max-width: 400px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.5);
}
#custom-tooltip.pinned {
  pointer-events: auto;
  overflow-y: auto;
  max-height: 80vh;
  border-color: #58a6ff;
  box-shadow: 0 4px 20px rgba(88,166,255,0.25);
}
</style>"""

TOOLTIP_DIV = '<div id="custom-tooltip"></div>\n'

# ---------------------------------------------------------------------------
# GLOBAL LEGEND HTML (visible by default; id="legend")
# ---------------------------------------------------------------------------

LEGEND_HTML = """<div id="legend" style="
    position:fixed; top:20px; right:20px; z-index:9999;
    background:rgba(13,17,23,0.92); border:1px solid #30363d;
    border-radius:8px; padding:14px 18px; font-family:sans-serif;
    font-size:13px; color:#e6edf3; min-width:200px; line-height:1.8">
  <div style="font-size:14px;font-weight:700;margin-bottom:8px">Global View</div>
  <div><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#4a90d9;margin-right:8px;vertical-align:middle"></span>West</div>
  <div><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#e05c5c;margin-right:8px;vertical-align:middle"></span>East</div>
  <div><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#a78bfa;margin-right:8px;vertical-align:middle"></span>Bridge</div>
  <hr style="border-color:#30363d;margin:10px 0">
  <div style="font-size:14px;font-weight:700;margin-bottom:8px">Edges</div>
  <div><span style="display:inline-block;width:18px;height:3px;background:#4caf50;margin-right:8px;vertical-align:middle;border-radius:2px"></span>Excitatory</div>
  <div><span style="display:inline-block;width:18px;height:3px;background:#cf6679;margin-right:8px;vertical-align:middle;border-radius:2px"></span>Inhibitory</div>
  <div><span style="display:inline-block;width:18px;height:3px;background:#f5a623;margin-right:8px;vertical-align:middle;border-radius:2px"></span>Mixed</div>
  <hr style="border-color:#30363d;margin:10px 0">
  <div style="color:#8b949e;font-size:11px">
    Node size = influence rank<br>
    Node color = hemisphere<br>
    Edge width = Hebbian connection strength<br>
    Hover nodes &amp; edges for detail
  </div>
</div>
"""

# ---------------------------------------------------------------------------
# SHARED HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def edge_color_from_valences(valences: list) -> str:
    """valences: list of (ps_id, name, v) where v is +1 or -1."""
    if not valences:
        return EXCITATORY_COLOR
    pos = sum(1 for _, _, v in valences if v > 0)
    neg = sum(1 for _, _, v in valences if v < 0)
    if neg == 0:
        return EXCITATORY_COLOR
    if pos == 0:
        return INHIBITORY_COLOR
    return MIXED_COLOR


def compute_hebbian_edge_weight(
    name_a: str, name_b: str, shared_ids: set, hebbian_weights: dict
) -> float:
    """
    Connection strength between two neurons across their shared phase sequences.
    Formula: sum of (w_a * w_b) for each shared PS.
    hebbian_weights: {neuron_name: {ps_id: float}}
    """
    wa = hebbian_weights.get(name_a, {})
    wb = hebbian_weights.get(name_b, {})
    return sum(wa.get(ps_id, 0.0) * wb.get(ps_id, 0.0) for ps_id in shared_ids)


def build_global_legend_html(ps_dominance: dict | None = None) -> str:
    """Global legend with optional PS dominance bar chart."""
    html = """<div id="legend" style="
    position:fixed; top:20px; right:20px; z-index:9999;
    background:rgba(13,17,23,0.92); border:1px solid #30363d;
    border-radius:8px; padding:14px 18px; font-family:sans-serif;
    font-size:13px; color:#e6edf3; min-width:220px; line-height:1.8">
  <div style="font-size:14px;font-weight:700;margin-bottom:8px">Global View</div>
  <div><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#4a90d9;margin-right:8px;vertical-align:middle"></span>West</div>
  <div><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#e05c5c;margin-right:8px;vertical-align:middle"></span>East</div>
  <div><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#a78bfa;margin-right:8px;vertical-align:middle"></span>Bridge</div>
  <hr style="border-color:#30363d;margin:10px 0">
  <div style="font-size:14px;font-weight:700;margin-bottom:8px">Edges</div>
  <div><span style="display:inline-block;width:18px;height:3px;background:#4caf50;margin-right:8px;vertical-align:middle;border-radius:2px"></span>Excitatory</div>
  <div><span style="display:inline-block;width:18px;height:3px;background:#cf6679;margin-right:8px;vertical-align:middle;border-radius:2px"></span>Inhibitory</div>
  <div><span style="display:inline-block;width:18px;height:3px;background:#f5a623;margin-right:8px;vertical-align:middle;border-radius:2px"></span>Mixed</div>
  <hr style="border-color:#30363d;margin:10px 0">
  <div style="color:#8b949e;font-size:11px">
    Node size = influence rank<br>
    Node color = hemisphere<br>
    Edge width = Hebbian connection strength<br>
    Hover nodes &amp; edges for detail
  </div>"""

    if ps_dominance:
        html += (
            "\n  <hr style='border-color:#30363d;margin:10px 0'>"
            "\n  <div style='font-size:12px;font-weight:700;margin-bottom:8px'>"
            "Phase Sequence Dominance</div>"
        )
        for ps_id, score in sorted(ps_dominance.items(), key=lambda x: -x[1]):
            bar_w = int(score * 80)
            color = "#4caf50" if score > 0.6 else "#f5a623" if score > 0.4 else "#8b949e"
            html += (
                f"\n  <div style='margin-top:5px'>"
                f"<span style='font-size:10px;color:#58a6ff;font-weight:600'>{ps_id}</span>"
                f"<div style='height:4px;background:#21262d;border-radius:2px;margin-top:3px'>"
                f"<div style='height:4px;width:{bar_w}px;max-width:100%;background:{color};"
                f"border-radius:2px'></div></div></div>"
            )

    html += "\n</div>\n"
    return html


def serve_and_open(output_path: str, port: int = 8766):
    directory = os.path.dirname(os.path.abspath(output_path))
    filename  = os.path.basename(output_path)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)
        def log_message(self, fmt, *args):
            pass   # suppress request logs

    server = http.server.HTTPServer(("", port), Handler)
    url    = f"http://localhost:{port}/{filename}"
    print(f"\nServing at {url}")
    webbrowser.open(url)
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.shutdown()


# ---------------------------------------------------------------------------
# GLOBAL NETWORK BUILDER
# ---------------------------------------------------------------------------

def global_node_size(rank: int, n_total: int) -> int:
    """Scale node size by rank, compressed for large models."""
    return max(8, int(40 - (rank - 1) * 32 / max(n_total - 1, 1)))


def build_network(
    global_data: dict,
    hebbian_state: dict | None = None,
):
    """
    Build the pyvis network for the global superorganism model.
    Returns (net, node_tooltips, edge_tooltips).
    Node color is hemisphere-based.
    """
    net = Network(
        height="100vh", width="100%",
        bgcolor=BG_COLOR, font_color=FONT_COLOR,
        directed=False, notebook=False,
    )

    people        = global_data["superorganism_list"]
    n_total       = len(people)
    node_tooltips = {}
    edge_tooltips = {}

    # --- Nodes ---
    for person in people:
        so        = person["superorganism"]
        hemisphere = so.get("hemisphere", "West")
        color     = HEMISPHERE_COLORS.get(hemisphere, HEMISPHERE_COLORS["West"])
        border    = HEMISPHERE_BORDER.get(hemisphere,  HEMISPHERE_BORDER["West"])
        size      = global_node_size(person["rank"], n_total)

        ps_rows = "".join(
            f"<div style='margin-top:6px'>"
            f"<span style='color:#58a6ff;font-weight:600;font-size:11px'>{ps['id']}</span>"
            f"&nbsp;<span style='color:#e6edf3'>{ps['name']}</span><br>"
            f"<span style='color:#8b949e;font-size:11px'>{ps.get('role','')}</span>"
            f"</div>"
            for ps in so.get("phase_sequences", [])
        )
        asm_items = "".join(
            f"<div style='margin-top:3px;color:#8b949e;font-size:11px'>"
            f"&middot; {a['name']}"
            f"<span style='color:#6e7681'> — {a.get('role','')}</span>"
            f"</div>"
            for a in so.get("cell_assemblies", [])
        )

        node_tooltips[person["name"]] = (
            f"<div style='max-width:360px;font-family:sans-serif;"
            f"font-size:13px;color:#e6edf3;line-height:1.5'>"
            f"<div style='font-size:16px;font-weight:700;margin-bottom:2px'>{person['name']}</div>"
            f"<div style='color:#8b949e;font-size:11px;margin-bottom:10px'>"
            f"{so.get('primary_sector','')} &nbsp;&middot;&nbsp; "
            f"<span style='color:{color}'>{hemisphere}</span></div>"
            f"<div style='color:#e6edf3;font-style:italic;margin-bottom:6px'>"
            f"{so.get('neuron_role','')}</div>"
            f"<div style='color:#8b949e;font-size:11px;margin-bottom:14px'>"
            f"{so.get('neuron_type','')}</div>"
            f"<div style='color:#e6edf3;font-weight:600;margin-bottom:2px'>Phase Sequences</div>"
            f"{ps_rows}"
            f"<div style='color:#e6edf3;font-weight:600;margin-top:14px;margin-bottom:2px'>"
            f"Cell Assemblies</div>"
            f"{asm_items}"
            f"</div>"
        )

        net.add_node(
            person["name"],
            label=person["name"],
            color={"background": color, "border": border,
                   "highlight": {"background": color, "border": "#ffffff"}},
            size=size,
            font={"size": 13, "color": FONT_COLOR, "face": "sans-serif"},
            borderWidth=2,
        )

    # --- Hebbian state setup ---
    hebbian_weights = None
    edge_threshold  = 0.0
    hebbian_week    = 0
    if hebbian_state:
        hebbian_weights = hebbian_state.get("neuron_dps_weights", {})
        edge_threshold  = hebbian_state.get("config", {}).get("edge_display_threshold", 0.0)
        hebbian_week    = hebbian_state.get("week_count", 0)

    # --- Edges ---
    for i, a in enumerate(people):
        for b in people[i + 1:]:
            ps_a = {ps["id"] for ps in a["superorganism"].get("phase_sequences", [])}
            ps_b = {ps["id"] for ps in b["superorganism"].get("phase_sequences", [])}
            shared_ids = ps_a & ps_b
            if not shared_ids:
                continue

            if hebbian_weights is not None:
                edge_weight = compute_hebbian_edge_weight(
                    a["name"], b["name"], shared_ids, hebbian_weights
                )
                if edge_weight < edge_threshold:
                    continue
                edge_value = edge_weight
            else:
                edge_weight = None
                edge_value  = len(shared_ids)

            # Global edges are all excitatory by default (no briefing-driven valences yet)
            ecolor = EXCITATORY_COLOR
            count  = len(shared_ids)

            if edge_weight is not None:
                strength_html = (
                    f"<div style='color:#8b949e;font-size:11px;margin-bottom:2px'>"
                    f"Hebbian strength: <span style='color:#e6edf3'>{edge_weight:.3f}</span>"
                    f" &nbsp;·&nbsp; week {hebbian_week}</div>"
                )
            else:
                strength_html = ""

            ps_lines = "".join(
                f"<div style='margin-top:5px'>"
                f"<span style='font-size:14px;color:#4caf50'>&#8853;</span>"
                f"&nbsp;<span style='color:#8b949e;font-size:11px'>{ps_id}</span>"
                f"</div>"
                for ps_id in sorted(shared_ids)
            )

            edge_key = f"{a['name']}|||{b['name']}"
            edge_tooltips[edge_key] = (
                f"<div style='font-family:sans-serif;font-size:13px;"
                f"color:#e6edf3;max-width:320px'>"
                f"<div style='font-weight:600;margin-bottom:4px'>"
                f"Shared phase sequences ({count})</div>"
                f"{strength_html}"
                f"{ps_lines}"
                f"</div>"
            )

            net.add_edge(
                a["name"], b["name"],
                value=edge_value,
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
# US-SPECIFIC CONSTANTS
# ---------------------------------------------------------------------------

SIGNAL_BORDER_COLOR = {
    "concerning": "#cf6679",
    "notable":    "#f5a623",
}

SIGNAL_ICON = {
    "concerning": ("▼", "#cf6679"),
    "notable":    ("▲", "#f5a623"),
    "quiet":      ("—", "#8b949e"),
}


def us_node_size(rank: int, n_total: int) -> int:
    """Scale node size for US model."""
    return max(8, int(40 - (rank - 1) * 32 / max(n_total - 1, 1)))


# ---------------------------------------------------------------------------
# LOADERS
# ---------------------------------------------------------------------------

def load_latest_briefing(script_dir: str) -> dict | None:
    briefings_dir = os.path.join(script_dir, "briefings")
    if not os.path.isdir(briefings_dir):
        return None
    files = sorted(
        (f for f in os.listdir(briefings_dir)
         if f.startswith("weekly_briefing_") and f.endswith(".json")
         and "_raw_" not in f),
        reverse=True,
    )
    if not files:
        return None
    path = os.path.join(briefings_dir, files[0])
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_edge_signals_map(briefing: dict) -> dict:
    result = {}
    for sig in briefing.get("edge_signals", []):
        key = frozenset({sig["person_a"], sig["person_b"]})
        if key not in result:
            result[key] = {}
        result[key][sig["ps_id"]] = (sig["valence"], sig.get("evidence", ""))
    return result


def load_hebbian_state(script_dir: str) -> dict | None:
    path = os.path.join(script_dir, "superorganism_state.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_global_hebbian_state(script_dir: str) -> dict | None:
    path = os.path.join(script_dir, "global_superorganism_state.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# LEGEND BUILDERS
# ---------------------------------------------------------------------------

def build_us_legend_html(dps_dominance: dict | None = None) -> str:
    html = """<div id="us-legend" style="
    display:none;
    position:fixed; top:20px; right:20px; z-index:9999;
    background:rgba(13,17,23,0.92); border:1px solid #30363d;
    border-radius:8px; padding:14px 18px; font-family:sans-serif;
    font-size:13px; color:#e6edf3; min-width:220px; line-height:1.8">
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
    Edge width = Hebbian connection strength<br>
    Hover nodes &amp; edges for detail
  </div>"""

    if dps_dominance:
        html += (
            "\n  <hr style='border-color:#30363d;margin:10px 0'>"
            "\n  <div style='font-size:12px;font-weight:700;margin-bottom:8px'>"
            "Phase Sequence Dominance</div>"
        )
        for dps_id, score in sorted(dps_dominance.items(), key=lambda x: -x[1]):
            bar_w = int(score * 80)
            color = "#4caf50" if score > 0.6 else "#f5a623" if score > 0.4 else "#8b949e"
            html += (
                f"\n  <div style='margin-top:5px'>"
                f"<span style='font-size:10px;color:#58a6ff;font-weight:600'>{dps_id}</span>"
                f"<div style='height:4px;background:#21262d;border-radius:2px;margin-top:3px'>"
                f"<div style='height:4px;width:{bar_w}px;max-width:100%;background:{color};"
                f"border-radius:2px'></div></div></div>"
            )

    html += "\n</div>\n"
    return html


# ---------------------------------------------------------------------------
# COMBINED-VIZ SPECIFIC HTML OVERLAYS
# ---------------------------------------------------------------------------

COMBINED_TITLE_HTML = """
<div id="graph-title-block" style="position:fixed;top:20px;left:20px;z-index:9999;font-family:sans-serif;color:#e6edf3">
  <div id="graph-title" style="font-size:20px;font-weight:bold;letter-spacing:0.5px">Human Superorganism</div>
  <div id="graph-subtitle" style="font-size:12px;color:#8b949e;margin-top:2px">Prime Movers</div>
</div>
"""

GLOBAL_ASM_LEGEND_HTML = """<div id="global-asm-legend" style="
    display:none;
    position:fixed; top:20px; right:20px; z-index:9999;
    background:rgba(13,17,23,0.92); border:1px solid #30363d;
    border-radius:8px; padding:14px 18px; font-family:sans-serif;
    font-size:13px; color:#e6edf3; min-width:200px; line-height:1.8">
  <div style="font-size:14px;font-weight:700;margin-bottom:8px">Global &middot; Assemblies</div>
  <div><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#4a90d9;margin-right:8px;vertical-align:middle"></span>West-dominated</div>
  <div><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#e05c5c;margin-right:8px;vertical-align:middle"></span>East-dominated</div>
  <div><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#a78bfa;margin-right:8px;vertical-align:middle"></span>Bridge</div>
  <hr style="border-color:#30363d;margin:10px 0">
  <div style="color:#8b949e;font-size:11px">
    Node size = member count<br>
    Edge = shared phase sequences<br>
    Hover for members &amp; news
  </div>
</div>
"""

US_ASM_LEGEND_HTML = """<div id="us-asm-legend" style="
    display:none;
    position:fixed; top:20px; right:20px; z-index:9999;
    background:rgba(13,17,23,0.92); border:1px solid #30363d;
    border-radius:8px; padding:14px 18px; font-family:sans-serif;
    font-size:13px; color:#e6edf3; min-width:200px; line-height:1.8">
  <div style="font-size:14px;font-weight:700;margin-bottom:8px">US &middot; Assemblies</div>
  <div><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#4a90d9;margin-right:8px;vertical-align:middle"></span>All nodes: West (US)</div>
  <hr style="border-color:#30363d;margin:10px 0">
  <div style="color:#8b949e;font-size:11px">
    Node size = member count<br>
    Edge = shared phase sequences<br>
    Hover for members &amp; news
  </div>
</div>
"""


def make_controls_html(
    n_global: int | None, n_us: int | None,
    n_global_asm: int | None, n_us_asm: int | None,
    has_global: bool,
) -> str:
    """Two-toggle control panel: scope (Global/US) + view type (Neurons/Assemblies)."""
    btn = ("color:#fff;border:none;border-radius:6px;padding:7px 16px;"
           "font-size:12px;font-weight:600;cursor:pointer;letter-spacing:0.3px;"
           "box-shadow:0 2px 6px rgba(0,0,0,0.4);")
    active_scope = "#238636"
    inactive     = "#21262d"
    active_type  = "#1f6feb"

    g_label  = f"Global ({n_global})"       if n_global    else "Global"
    u_label  = f"US ({n_us})"               if n_us        else "US"

    scope_row = ""
    if has_global:
        scope_row = (
            f"<div style='display:flex;gap:6px;margin-bottom:6px'>"
            f"<button id='btn-scope-global' onclick=\"switchView('global',currentViewType)\""
            f" style='{btn}background:{active_scope}'>{g_label}</button>"
            f"<button id='btn-scope-us' onclick=\"switchView('us',currentViewType)\""
            f" style='{btn}background:{inactive}'>{u_label}</button>"
            f"</div>"
        )

    type_row = (
        f"<div style='display:flex;gap:6px'>"
        f"<button id='btn-type-neuron' onclick=\"switchView(currentScope,'neuron')\""
        f" style='{btn}background:{active_type}'>Neurons</button>"
        f"<button id='btn-type-assembly' onclick=\"switchView(currentScope,'assembly')\""
        f" style='{btn}background:{inactive}'>Assemblies</button>"
        f"</div>"
    )

    return (
        "<div id='view-controls' style='position:fixed;top:20px;left:50%;"
        "transform:translateX(-50%);z-index:9999;font-family:sans-serif;"
        "display:flex;flex-direction:column;align-items:center'>"
        f"{scope_row}{type_row}"
        "</div>"
    )


# ---------------------------------------------------------------------------
# US NETWORK BUILDER
# ---------------------------------------------------------------------------

def compute_us_ps_valences(
    person_a: dict, person_b: dict, shared_ids: set,
    edge_signals_map: dict | None = None,
) -> list:
    pair_key     = frozenset({person_a["name"], person_b["name"]})
    pair_signals = (edge_signals_map or {}).get(pair_key, {})
    ps_map       = {ps["id"]: ps["name"]
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


def build_us_network(
    us_data: dict,
    briefing: dict | None = None,
    hebbian_state: dict | None = None,
):
    """
    Build the pyvis network for the US superorganism model.
    Returns (net, node_tooltips, edge_tooltips).
    """
    net = Network(
        height="100vh", width="100%",
        bgcolor=BG_COLOR, font_color=FONT_COLOR,
        directed=False, notebook=False,
    )

    people        = us_data["superorganism_list"]
    n_total       = len(people)
    node_tooltips = {}
    edge_tooltips = {}

    color  = HEMISPHERE_COLORS["West"]
    border = HEMISPHERE_BORDER["West"]

    edge_signals_map  = build_edge_signals_map(briefing) if briefing else {}
    person_signals    = {}
    person_summaries  = {}
    person_top_stories: dict = {}
    ps_updates_map: dict     = {}
    briefing_week    = ""
    if briefing:
        briefing_week = briefing.get("week_ending", "")
        for pu in briefing.get("person_updates", []):
            person_signals[pu["name"]]   = pu.get("signal", "quiet")
            person_summaries[pu["name"]] = pu.get("summary", "")
        for story in briefing.get("top_stories", []):
            for name in story.get("persons", []):
                person_top_stories.setdefault(name, []).append(story)
        ps_updates_map = {u["id"]: u for u in briefing.get("phase_sequence_updates", [])}

    _m_icon  = {"accelerating": "↑", "stable": "→", "decelerating": "↓"}
    _m_color = {"accelerating": "#4caf50", "stable": "#8b949e", "decelerating": "#cf6679"}

    # --- Nodes ---
    for person in people:
        so   = person["superorganism"]
        size = us_node_size(person["rank"], n_total)
        ps_rows = ""
        for ps in so.get("phase_sequences", []):
            ps_upd   = ps_updates_map.get(ps["id"], {})
            momentum = ps_upd.get("momentum", "")
            mom_html = (
                f" <span style='color:{_m_color[momentum]}'>{_m_icon[momentum]}</span>"
                if momentum in _m_icon else ""
            )
            ps_rows += (
                f"<div style='margin-top:6px'>"
                f"<span style='color:#58a6ff;font-weight:600;font-size:11px'>{ps['id']}</span>"
                f"&nbsp;<span style='color:#e6edf3'>{ps['name']}</span>{mom_html}<br>"
                f"<span style='color:#8b949e;font-size:11px'>{ps.get('role','')}</span>"
                f"</div>"
            )

        asm_items = "".join(
            f"<div style='margin-top:3px;color:#8b949e;font-size:11px'>"
            f"&middot; {a['name']}"
            f"<span style='color:#6e7681'> — {a.get('role','')}</span>"
            f"</div>"
            for a in so.get("cell_assemblies", [])
        )

        sig         = person_signals.get(person["name"], "quiet")
        node_border = SIGNAL_BORDER_COLOR.get(sig, border)
        sig_icon, sig_color = SIGNAL_ICON.get(sig, ("—", "#8b949e"))

        weekly_html    = ""
        weekly_summary = person_summaries.get(person["name"], "")
        if weekly_summary:
            week_label  = f"This Week ({briefing_week})" if briefing_week else "This Week"
            weekly_html = (
                f"<hr style='border-color:#30363d;margin:10px 0'>"
                f"<div style='font-size:12px;font-weight:600;color:{sig_color};"
                f"margin-bottom:4px'>{sig_icon} {week_label}</div>"
                f"<div style='color:#8b949e;font-size:11px'>{weekly_summary}</div>"
            )
        elif briefing:
            weekly_html = (
                "<hr style='border-color:#30363d;margin:10px 0'>"
                "<div style='font-size:11px;color:#6e7681'>Not tracked this week</div>"
            )

        stories = person_top_stories.get(person["name"], [])
        if stories:
            weekly_html += (
                "<hr style='border-color:#30363d;margin:10px 0'>"
                "<div style='font-size:12px;font-weight:600;margin-bottom:4px'>Top Stories</div>"
            )
            for story in stories[:3]:
                valence = story.get("valence", "neutral")
                v_color = (
                    "#4caf50" if valence == "cooperative"
                    else "#cf6679" if valence == "adversarial"
                    else "#8b949e"
                )
                ps_tag = (
                    f" <span style='color:#58a6ff;font-size:10px'>{story['ps_id']}</span>"
                    if story.get("ps_id") else ""
                )
                weekly_html += (
                    f"<div style='margin-top:6px'>"
                    f"<span style='color:{v_color};font-size:11px;font-weight:600'>"
                    f"{story['headline']}</span>{ps_tag}<br>"
                    f"<span style='color:#8b949e;font-size:10px'>{story['significance']}</span>"
                    f"</div>"
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

    # --- Hebbian state setup ---
    hebbian_weights = None
    edge_threshold  = 0.0
    hebbian_week    = 0
    if hebbian_state:
        hebbian_weights = hebbian_state.get("neuron_dps_weights", {})
        edge_threshold  = hebbian_state.get("config", {}).get("edge_display_threshold", 0.0)
        hebbian_week    = hebbian_state.get("week_count", 0)

    # --- Edges ---
    for i, a in enumerate(people):
        for b in people[i + 1:]:
            ps_a = {ps["id"] for ps in a["superorganism"].get("phase_sequences", [])}
            ps_b = {ps["id"] for ps in b["superorganism"].get("phase_sequences", [])}
            shared_ids = ps_a & ps_b
            if not shared_ids:
                continue

            if hebbian_weights is not None:
                edge_weight = compute_hebbian_edge_weight(
                    a["name"], b["name"], shared_ids, hebbian_weights
                )
                if edge_weight < edge_threshold:
                    continue
                edge_value = edge_weight
            else:
                edge_weight = None
                edge_value  = len(shared_ids)

            valences    = compute_us_ps_valences(a, b, shared_ids, edge_signals_map)
            ecolor      = edge_color_from_valences(valences)
            count       = len(valences)
            excit_count = sum(1 for _, _, v in valences if v > 0)
            inhib_count = sum(1 for _, _, v in valences if v < 0)

            pair_key          = frozenset({a["name"], b["name"]})
            pair_signals_data = edge_signals_map.get(pair_key, {})

            if edge_weight is not None:
                strength_html = (
                    f"<div style='color:#8b949e;font-size:11px;margin-bottom:2px'>"
                    f"Hebbian strength: "
                    f"<span style='color:#e6edf3'>{edge_weight:.3f}</span>"
                    f" &nbsp;·&nbsp; week {hebbian_week}</div>"
                )
            else:
                strength_html = ""

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
                ev_html  = (
                    f"<div style='color:#8b949e;font-size:10px;font-style:italic;"
                    f"padding-left:18px;margin-top:1px'>{evidence}</div>"
                    if evidence else ""
                )
                sym = "&#8853;" if v > 0 else "&#8854;"
                c   = "#4caf50" if v > 0 else "#cf6679"
                ps_lines += (
                    f"<div style='margin-top:5px'>"
                    f"<span style='font-size:14px;color:{c}'>{sym}</span>"
                    f"&nbsp;<span style='color:#8b949e;font-size:11px'>{pid}</span>"
                    f"&nbsp;{name}"
                    f"{ev_html}"
                    f"</div>"
                )

            edge_key = f"{a['name']}|||{b['name']}"
            edge_tooltips[edge_key] = (
                f"<div style='font-family:sans-serif;font-size:13px;color:#e6edf3;"
                f"max-width:320px'>"
                f"<div style='font-weight:600;margin-bottom:4px'>"
                f"Shared phase sequences ({count})</div>"
                f"{strength_html}"
                f"<div style='font-size:11px;margin-bottom:8px'>{valence_label}</div>"
                f"{ps_lines}"
                f"</div>"
            )

            net.add_edge(
                a["name"], b["name"],
                value=edge_value,
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
# ASSEMBLY NETWORK BUILDER (shared for US and global)
# ---------------------------------------------------------------------------

def build_assembly_network(
    model_data: dict,
    briefing: dict | None = None,
    is_us: bool = True,
):
    """
    Build a pyvis network where nodes are cell assemblies.
    Node size = member count. Edges = shared PS membership.
    Returns (net, node_tooltips, edge_tooltips).
    """
    net = Network(
        height="100vh", width="100%",
        bgcolor=BG_COLOR, font_color=FONT_COLOR,
        directed=False, notebook=False,
    )

    raw_assemblies = model_data.get("canonical_vocabulary", {}).get("cell_assemblies", [])
    people         = model_data["superorganism_list"]

    # Normalize canonical assemblies: may be strings or dicts
    assemblies = []
    for ca in raw_assemblies:
        if isinstance(ca, str):
            assemblies.append({"id": ca, "name": ca, "ps_memberships": []})
        else:
            assemblies.append(ca)

    # ca_id → list of person dicts that are members
    # Person-level cell_assemblies may have no "id" — fall back to "name"
    ca_members: dict = {}
    for person in people:
        for ca in person.get("superorganism", {}).get("cell_assemblies", []):
            ca_id = ca.get("id") or ca.get("name", "")
            if ca_id:
                ca_members.setdefault(ca_id, []).append(person)

    # assembly_updates from briefing (keyed by id)
    asm_updates: dict = {}
    if briefing:
        for au in briefing.get("assembly_updates", []):
            asm_updates[au["id"]] = au

    node_tooltips: dict = {}
    edge_tooltips: dict = {}

    max_members = max((len(ca_members.get(ca["id"], [])) for ca in assemblies), default=1)

    for ca in assemblies:
        members  = ca_members.get(ca["id"], [])
        n_members = len(members)
        size     = max(10, int(10 + 30 * n_members / max(max_members, 1)))

        if is_us:
            color  = HEMISPHERE_COLORS["West"]
            border = HEMISPHERE_BORDER["West"]
        else:
            # Color by majority hemisphere of members
            hemi_counts: dict = {}
            for p in members:
                h = p["superorganism"].get("hemisphere", "West")
                hemi_counts[h] = hemi_counts.get(h, 0) + 1
            dominant = max(hemi_counts, key=hemi_counts.get) if hemi_counts else "Global"
            color  = HEMISPHERE_COLORS.get(dominant, HEMISPHERE_COLORS["West"])
            border = HEMISPHERE_BORDER.get(dominant, HEMISPHERE_BORDER["West"])

        ps_list = ca.get("ps_memberships", [])
        ps_html = "".join(
            f"<div style='margin-top:4px'>"
            f"<span style='color:#58a6ff;font-weight:600;font-size:11px'>{pid}</span>"
            f"</div>"
            for pid in ps_list
        )
        member_html = "".join(
            f"<div style='color:#8b949e;font-size:11px;margin-top:2px'>"
            f"&middot; {p['name']}</div>"
            for p in sorted(members, key=lambda x: x["rank"])
        )

        upd = asm_updates.get(ca["id"], {})
        update_html = ""
        if upd.get("summary"):
            sig       = upd.get("signal", "active")
            sig_color = "#4caf50" if sig == "active" else "#8b949e"
            sig_icon  = "▲" if sig == "active" else "—"
            update_html = (
                f"<hr style='border-color:#30363d;margin:10px 0'>"
                f"<div style='font-size:12px;font-weight:600;color:{sig_color};"
                f"margin-bottom:4px'>{sig_icon} This Week</div>"
                f"<div style='color:#8b949e;font-size:11px'>{upd['summary']}</div>"
            )

        node_tooltips[ca["id"]] = (
            f"<div style='max-width:340px;font-family:sans-serif;"
            f"font-size:13px;color:#e6edf3;line-height:1.5'>"
            f"<div style='font-size:16px;font-weight:700;margin-bottom:2px'>{ca['name']}</div>"
            f"<div style='color:#8b949e;font-size:11px;margin-bottom:10px'>"
            f"{ca.get('role', ca.get('description', ''))}</div>"
            f"<div style='color:#e6edf3;font-weight:600;margin-bottom:4px'>Phase Sequences</div>"
            f"{ps_html}"
            f"<div style='color:#e6edf3;font-weight:600;margin-top:10px;margin-bottom:4px'>"
            f"Members ({n_members})</div>"
            f"{member_html}"
            f"{update_html}"
            f"</div>"
        )

        net.add_node(
            ca["id"],
            label=ca["name"],
            color={"background": color, "border": border,
                   "highlight": {"background": color, "border": "#ffffff"}},
            size=size,
            font={"size": 12, "color": FONT_COLOR, "face": "sans-serif"},
            borderWidth=2,
        )

    # Edges: shared PS memberships
    for i, ca_a in enumerate(assemblies):
        for ca_b in assemblies[i + 1:]:
            ps_a   = set(ca_a.get("ps_memberships", []))
            ps_b   = set(ca_b.get("ps_memberships", []))
            shared = ps_a & ps_b
            if not shared:
                continue

            shared_members = (
                {p["name"] for p in ca_members.get(ca_a["id"], [])} &
                {p["name"] for p in ca_members.get(ca_b["id"], [])}
            )
            ps_lines = "".join(
                f"<div style='margin-top:4px'>"
                f"<span style='color:#58a6ff;font-size:11px'>{pid}</span></div>"
                for pid in sorted(shared)
            )
            mem_lines = "".join(
                f"<div style='color:#8b949e;font-size:11px;margin-top:2px'>"
                f"&middot; {n}</div>"
                for n in sorted(shared_members)
            )
            edge_key = f"{ca_a['id']}|||{ca_b['id']}"
            edge_tooltips[edge_key] = (
                f"<div style='font-family:sans-serif;font-size:13px;"
                f"color:#e6edf3;max-width:280px'>"
                f"<div style='font-weight:600;margin-bottom:4px'>"
                f"Shared phase sequences ({len(shared)})</div>"
                f"{ps_lines}"
                + (f"<div style='font-weight:600;margin-top:8px;margin-bottom:4px'>"
                   f"Shared members</div>{mem_lines}" if mem_lines else "")
                + f"</div>"
            )

            net.add_edge(
                ca_a["id"], ca_b["id"],
                value=len(shared),
                color={"color": EXCITATORY_COLOR, "highlight": "#ffffff", "opacity": 0.75},
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
        "scaling": { "min": 1, "max": 6 }
      },
      "physics": {
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -80,
          "centralGravity": 0.005,
          "springLength": 150,
          "springConstant": 0.08,
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
    lines     = html.splitlines(keepends=True)
    new_lines = []
    for line in lines:
        stripped = line.strip()
        indent   = line[: len(line) - len(line.lstrip())]
        if stripped.startswith("nodes = new vis.DataSet("):
            new_lines.append(indent + "nodes = globalNeuronNodes;\n")
        elif stripped.startswith("edges = new vis.DataSet("):
            new_lines.append(indent + "edges = globalNeuronEdges;\n")
        else:
            new_lines.append(line)
    return "".join(new_lines)


def build_combined_html(
    # Global neuron (default view, used as the HTML base)
    global_net, global_node_tooltips, global_edge_tooltips,
    output_path: str,
    # US neuron
    us_net=None, us_node_tooltips=None, us_edge_tooltips=None,
    # Global assembly
    global_asm_net=None, global_asm_node_tooltips=None, global_asm_edge_tooltips=None,
    # US assembly
    us_asm_net=None, us_asm_node_tooltips=None, us_asm_edge_tooltips=None,
    # Legends
    global_legend_html=None, us_legend_html=None,
    global_asm_legend_html=None, us_asm_legend_html=None,
    # Counts for control labels
    n_global=None, n_us=None, n_global_asm=None, n_us_asm=None,
    has_global: bool = True,
):
    """
    Write combined HTML supporting up to 4 views via two toggles:
      scope toggle  : Global / US
      view toggle   : Neurons / Assemblies
    global_net is always the primary (its HTML is used as the page base).
    """
    # --- Write all nets to temp files and read back ---
    nets = {
        "globalNeuron": global_net,
        "usNeuron":     us_net,
        "globalAsm":    global_asm_net,
        "usAsm":        us_asm_net,
    }
    tmp_files = {k: output_path + f".{k}.tmp.html" for k in nets if nets[k]}
    html_bodies: dict = {}

    try:
        for key, net in nets.items():
            if net:
                net.write_html(tmp_files[key])
        for key in tmp_files:
            with open(tmp_files[key], "r", encoding="utf-8") as f:
                html_bodies[key] = f.read()
    finally:
        for p in tmp_files.values():
            if os.path.exists(p):
                os.remove(p)

    # Base HTML comes from global neuron net; replace its DataSet lines
    html = replace_dataset_lines(html_bodies["globalNeuron"])

    # --- Extract dataset JSON from each view ---
    def ds_block(key, nodes_var, edges_var):
        if key not in html_bodies:
            return (
                f"var {nodes_var} = new vis.DataSet([]);\n"
                f"var {edges_var} = new vis.DataSet([]);\n"
            )
        n_json, e_json = extract_dataset_json(html_bodies[key])
        return (
            f"var {nodes_var}Data = {n_json};\n"
            f"var {edges_var}Data = {e_json};\n"
            f"var {nodes_var} = new vis.DataSet({nodes_var}Data);\n"
            f"var {edges_var} = new vis.DataSet({edges_var}Data);\n"
        )

    dataset_script = (
        "<script type=\"text/javascript\">\n"
        + ds_block("globalNeuron", "globalNeuronNodes", "globalNeuronEdges")
        + ds_block("usNeuron",     "usNeuronNodes",     "usNeuronEdges")
        + ds_block("globalAsm",    "globalAsmNodes",    "globalAsmEdges")
        + ds_block("usAsm",        "usAsmNodes",        "usAsmEdges")
        + "var currentScope = 'global';\n"
        + "var currentViewType = 'neuron';\n"
        + "</script>\n"
    )
    html = html.replace("</head>", TOOLTIP_CSS + dataset_script + "</head>", 1)

    # --- Tooltip dicts for all 4 views ---
    def tip_js(var, d):
        return f"var {var} = {json.dumps(d or {}, ensure_ascii=False)};\n"

    tooltip_dicts_js = (
        "<script type=\"text/javascript\">\n"
        + tip_js("globalNeuronNodeTooltips", global_node_tooltips)
        + tip_js("globalNeuronEdgeTooltips", global_edge_tooltips)
        + tip_js("usNeuronNodeTooltips",     us_node_tooltips)
        + tip_js("usNeuronEdgeTooltips",     us_edge_tooltips)
        + tip_js("globalAsmNodeTooltips",    global_asm_node_tooltips)
        + tip_js("globalAsmEdgeTooltips",    global_asm_edge_tooltips)
        + tip_js("usAsmNodeTooltips",        us_asm_node_tooltips)
        + tip_js("usAsmEdgeTooltips",        us_asm_edge_tooltips)
        + "var activeNodeTooltips = globalNeuronNodeTooltips;\n"
        + "var activeEdgeTooltips = globalNeuronEdgeTooltips;\n"
        + "window.updateTooltipMap = function(scope, viewType) {\n"
        + "  var maps = {\n"
        + "    'global_neuron':   [globalNeuronNodeTooltips, globalNeuronEdgeTooltips],\n"
        + "    'us_neuron':       [usNeuronNodeTooltips,     usNeuronEdgeTooltips],\n"
        + "    'global_assembly': [globalAsmNodeTooltips,    globalAsmEdgeTooltips],\n"
        + "    'us_assembly':     [usAsmNodeTooltips,        usAsmEdgeTooltips],\n"
        + "  };\n"
        + "  var m = maps[scope + '_' + viewType] || maps['global_neuron'];\n"
        + "  activeNodeTooltips = m[0]; activeEdgeTooltips = m[1];\n"
        + "};\n"
        + "window.getActiveNodeTooltips = function() { return activeNodeTooltips; };\n"
        + "window.getActiveEdgeTooltips = function() { return activeEdgeTooltips; };\n"
        + "</script>\n"
    )

    tooltip_listener_js = """<script type="text/javascript">
(function() {
  var mouseX = 0, mouseY = 0;
  var pinnedNodeId = null;

  document.addEventListener('mousemove', function(e) {
    mouseX = e.clientX; mouseY = e.clientY;
    var tip = document.getElementById('custom-tooltip');
    if (tip && tip.style.display === 'block' && !pinnedNodeId) positionTooltip(tip);
  });
  function positionTooltip(tip) {
    var pad = 18, tw = tip.offsetWidth, th = tip.offsetHeight;
    var vw = window.innerWidth, vh = window.innerHeight;
    var x = mouseX + pad, y = mouseY - pad;
    if (x + tw > vw - pad) x = mouseX - tw - pad;
    if (y + th > vh - pad) y = vh - th - pad;
    if (y < pad) y = pad;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  }
  function pinTooltip(tip) {
    var pad = 18, tw = tip.offsetWidth, th = tip.offsetHeight;
    var vw = window.innerWidth, vh = window.innerHeight;
    var x = mouseX + pad, y = mouseY - pad;
    if (x + tw > vw - pad) x = mouseX - tw - pad;
    if (y + th > vh - pad) y = vh - th - pad;
    if (y < pad) y = pad;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
    tip.classList.add('pinned');
  }
  function showTooltip(html) {
    var tip = document.getElementById('custom-tooltip');
    tip.innerHTML = html; tip.style.display = 'block'; positionTooltip(tip);
  }
  function hideTooltip() {
    var tip = document.getElementById('custom-tooltip');
    if (tip) { tip.style.display = 'none'; tip.classList.remove('pinned'); }
    pinnedNodeId = null;
  }
  var pollCount = 0;
  var poll = setInterval(function() {
    pollCount++;
    if (typeof network !== 'undefined' && network.body && network.on) {
      clearInterval(poll); attachListeners();
    }
    if (pollCount > 100) clearInterval(poll);
  }, 50);
  function attachListeners() {
    network.on('hoverNode', function(p) {
      if (pinnedNodeId !== null) return;
      var html = window.getActiveNodeTooltips()[p.node];
      if (html) showTooltip(html);
    });
    network.on('blurNode', function() {
      if (pinnedNodeId !== null) return;
      hideTooltip();
    });
    network.on('hoverEdge', function(p) {
      if (pinnedNodeId !== null) return;
      var edge = network.body.data.edges.get(p.edge);
      if (!edge) return;
      var tips = window.getActiveEdgeTooltips();
      var html = tips[edge.from + '|||' + edge.to] || tips[edge.to + '|||' + edge.from];
      if (html) showTooltip(html);
    });
    network.on('blurEdge', function() {
      if (pinnedNodeId !== null) return;
      hideTooltip();
    });
    network.on('dragStart', function() {
      if (pinnedNodeId !== null) return;
      hideTooltip();
    });
    network.on('click', function(p) {
      var tip = document.getElementById('custom-tooltip');
      if (p.nodes && p.nodes.length > 0) {
        var clickedId = p.nodes[0];
        if (pinnedNodeId === clickedId) {
          network.unselectAll();
          hideTooltip();
        } else {
          pinnedNodeId = clickedId;
          var html = window.getActiveNodeTooltips()[clickedId];
          if (html) { tip.innerHTML = html; tip.style.display = 'block'; pinTooltip(tip); }
        }
      } else {
        if (pinnedNodeId !== null) hideTooltip();
      }
    });
  }
})();
</script>
"""

    switch_view_js = """<script type="text/javascript">
window.switchView = function(scope, viewType) {
  currentScope    = scope;
  currentViewType = viewType;
  var key = scope + '_' + viewType;
  var datasets = {
    'global_neuron':   {nodes: globalNeuronNodes, edges: globalNeuronEdges},
    'us_neuron':       {nodes: usNeuronNodes,     edges: usNeuronEdges},
    'global_assembly': {nodes: globalAsmNodes,    edges: globalAsmEdges},
    'us_assembly':     {nodes: usAsmNodes,        edges: usAsmEdges},
  };
  var ds = datasets[key];
  if (!ds) return;
  network.setData(ds);
  network.stabilize(150);
  window.updateTooltipMap(scope, viewType);
  allNodes = ds.nodes.get({returnType: 'Object'});
  allEdges = ds.edges.get({returnType: 'Object'});
  nodeColors = {};
  for (var nid in allNodes) { nodeColors[nid] = allNodes[nid].color; }

  // Update scope buttons
  ['global','us'].forEach(function(s) {
    var btn = document.getElementById('btn-scope-' + s);
    if (btn) btn.style.background = (s === scope) ? '#238636' : '#21262d';
  });
  // Update type buttons
  ['neuron','assembly'].forEach(function(t) {
    var btn = document.getElementById('btn-type-' + t);
    if (btn) btn.style.background = (t === viewType) ? '#1f6feb' : '#21262d';
  });
  // Update title
  var titles = {
    'global_neuron':   ['Human Superorganism', 'Global \u00b7 Neurons'],
    'us_neuron':       ['US Superorganism',    'US \u00b7 Neurons'],
    'global_assembly': ['Human Superorganism', 'Global \u00b7 Assemblies'],
    'us_assembly':     ['US Superorganism',    'US \u00b7 Assemblies'],
  };
  var t = titles[key] || titles['global_neuron'];
  var titleEl    = document.getElementById('graph-title');
  var subtitleEl = document.getElementById('graph-subtitle');
  if (titleEl)    titleEl.textContent    = t[0];
  if (subtitleEl) subtitleEl.textContent = t[1];
  // Show/hide legends
  var legendMap = {
    'global_neuron':   'legend',
    'us_neuron':       'us-legend',
    'global_assembly': 'global-asm-legend',
    'us_assembly':     'us-asm-legend',
  };
  ['legend','us-legend','global-asm-legend','us-asm-legend'].forEach(function(id) {
    var el = document.getElementById(id);
    if (el) el.style.display = (id === legendMap[key]) ? 'block' : 'none';
  });
};
</script>
"""

    controls_html = make_controls_html(n_global, n_us, n_global_asm, n_us_asm, has_global)

    body_injection = (
        TOOLTIP_DIV
        + controls_html
        + (us_legend_html     or "")
        + (global_legend_html or LEGEND_HTML)
        + (global_asm_legend_html or GLOBAL_ASM_LEGEND_HTML)
        + (us_asm_legend_html     or US_ASM_LEGEND_HTML)
        + COMBINED_TITLE_HTML
        + tooltip_dicts_js
        + tooltip_listener_js
        + switch_view_js
    )
    html = html.replace("</body>", body_injection + "</body>", 1)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    global_path = os.path.join(script_dir, "superorganism_model.json")
    us_path     = os.path.join(script_dir, "us_superorganism_model.json")
    output_path = os.path.join(script_dir, "combined_viz.html")

    print("=" * 60)
    print("COMBINED SUPERORGANISM VISUALIZER")
    print("=" * 60)

    if not os.path.exists(us_path):
        print(f"\nERROR: us_superorganism_model.json not found.")
        print("  Run: python superorganism_assembler.py --scope us")
        return

    with open(us_path, "r", encoding="utf-8") as f:
        us_data = json.load(f)

    has_global = os.path.exists(global_path)
    global_data = None
    if has_global:
        with open(global_path, "r", encoding="utf-8") as f:
            global_data = json.load(f)
    else:
        print("\nNote: superorganism_model.json not found — running US-only mode.")
        print("  Run global pipeline to enable toggle:")
        print("  python ps_council.py --scope global")
        print("  python ca_council.py --scope global")
        print("  python superorganism_assembler.py --scope global")

    n_us     = len(us_data["superorganism_list"])
    n_global = len(global_data["superorganism_list"]) if global_data else None
    print(f"\nUS model:     {n_us} nodes")
    if n_global:
        print(f"Global model: {n_global} nodes")

    # --- Global network (primary view) ---
    global_net = global_node_tooltips = global_edge_tooltips = None
    global_legend = None

    if global_data:
        print("\nLoading global Hebbian state...")
        global_hebbian = load_global_hebbian_state(script_dir)
        if global_hebbian:
            print(f"  State: week {global_hebbian.get('week_count', 0)}, "
                  f"last updated {global_hebbian.get('last_updated', '?')}")
        else:
            print("  No state — run: python hebbian_updater.py --bootstrap --scope global")

        print("Building global network...")
        global_net, global_node_tooltips, global_edge_tooltips = build_network(
            global_data, hebbian_state=global_hebbian
        )
        ps_dom       = global_hebbian.get("dps_dominance") if global_hebbian else None
        global_legend = build_global_legend_html(ps_dom)

    # --- US network ---
    print("\nLoading latest US briefing...")
    briefing = load_latest_briefing(script_dir)
    if briefing:
        n_sig = len(briefing.get("edge_signals", []))
        print(f"  Week ending {briefing.get('week_ending','?')} "
              f"({n_sig} edge signal{'s' if n_sig != 1 else ''})")
    else:
        print("  No briefing found — edges default to excitatory")

    print("Loading US Hebbian state...")
    hebbian_state = load_hebbian_state(script_dir)
    if hebbian_state:
        print(f"  State: week {hebbian_state.get('week_count', 0)}, "
              f"last updated {hebbian_state.get('last_updated', '?')}")
    else:
        print("  No state — run: python hebbian_updater.py --bootstrap --scope us")

    print("Building US neuron network...")
    us_net, us_node_tooltips, us_edge_tooltips = build_us_network(
        us_data, briefing=briefing, hebbian_state=hebbian_state
    )
    dps_dom   = hebbian_state.get("dps_dominance") if hebbian_state else None
    us_legend = build_us_legend_html(dps_dom)

    # --- Assembly networks ---
    print("Building US assembly network...")
    us_asm_net, us_asm_node_tooltips, us_asm_edge_tooltips = build_assembly_network(
        us_data, briefing=briefing, is_us=True
    )
    n_us_asm = len(us_data.get("canonical_vocabulary", {}).get("cell_assemblies", []))

    global_asm_net = global_asm_node_tooltips = global_asm_edge_tooltips = None
    n_global_asm   = None
    if global_data:
        print("Building global assembly network...")
        global_asm_net, global_asm_node_tooltips, global_asm_edge_tooltips = build_assembly_network(
            global_data, briefing=None, is_us=False
        )
        n_global_asm = len(global_data.get("canonical_vocabulary", {}).get("cell_assemblies", []))

    # --- Assemble HTML ---
    print("\nAssembling HTML...")

    if global_net:
        build_combined_html(
            global_net=global_net,
            global_node_tooltips=global_node_tooltips,
            global_edge_tooltips=global_edge_tooltips,
            output_path=output_path,
            us_net=us_net,
            us_node_tooltips=us_node_tooltips,
            us_edge_tooltips=us_edge_tooltips,
            global_asm_net=global_asm_net,
            global_asm_node_tooltips=global_asm_node_tooltips,
            global_asm_edge_tooltips=global_asm_edge_tooltips,
            us_asm_net=us_asm_net,
            us_asm_node_tooltips=us_asm_node_tooltips,
            us_asm_edge_tooltips=us_asm_edge_tooltips,
            global_legend_html=global_legend,
            us_legend_html=us_legend,
            n_global=n_global,
            n_us=n_us,
            n_global_asm=n_global_asm,
            n_us_asm=n_us_asm,
            has_global=True,
        )
    else:
        # US-only: scope toggle hidden, view type toggle still shown
        us_legend_visible = us_legend.replace('display:none;', '', 1)
        build_combined_html(
            global_net=us_net,
            global_node_tooltips=us_node_tooltips,
            global_edge_tooltips=us_edge_tooltips,
            output_path=output_path,
            us_asm_net=us_asm_net,
            us_asm_node_tooltips=us_asm_node_tooltips,
            us_asm_edge_tooltips=us_asm_edge_tooltips,
            global_legend_html=us_legend_visible,
            n_global=n_us,
            n_us_asm=n_us_asm,
            has_global=False,
        )

    print(f"Written to {output_path}")
    serve_and_open(output_path, port=8766)


if __name__ == "__main__":
    main()
