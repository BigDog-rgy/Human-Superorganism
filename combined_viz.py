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

import argparse
import json
import math
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
#loadingBar { display: none !important; }
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


def build_global_legend_html(briefing: dict | None = None) -> str:
    """Global legend with optional weekly news feed."""
    html = """<div id="legend" style="
    position:fixed; top:20px; right:20px; z-index:9999;
    background:rgba(13,17,23,0.92); border:1px solid #30363d;
    border-radius:8px; padding:14px 18px; font-family:sans-serif;
    font-size:13px; color:#e6edf3; min-width:220px; max-width:320px; line-height:1.8">
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

    if briefing:
        meta = briefing.get("_metadata", {})
        conscious_set   = set(meta.get("neurons_conscious", []))
        spontaneous_set = set(meta.get("neurons_spontaneous", []))

        ps_updates = briefing.get("phase_sequence_updates", [])
        person_updates = briefing.get("person_updates", [])
        conscious_updates = [
            p for p in person_updates
            if p["name"] in conscious_set and p.get("signal", "active") == "active"
        ]
        spontaneous_updates = [
            p for p in person_updates
            if p["name"] in spontaneous_set and p.get("signal", "active") == "active"
        ]

        def news_item(label: str, summary: str) -> str:
            safe_label   = label.replace('"', "&quot;").replace("'", "&#39;") if label else ""
            safe_summary = summary.replace('"', "&quot;").replace("'", "&#39;") if summary else ""
            preview  = summary[:120] + ("…" if len(summary) > 120 else "") if summary else ""
            clickable = bool(summary)
            cursor   = "pointer" if clickable else "default"
            onclick  = ' onclick="openNewsModal(this)"' if clickable else ""
            hover_style = (
                " onmouseover=\"this.style.background='rgba(88,166,255,0.06)'\" "
                "onmouseout=\"this.style.background='rgba(255,255,255,0.03)'\""
                if clickable else ""
            )
            return (
                f"<div style='margin-bottom:6px;padding:4px 6px;border-left:3px solid #58a6ff;"
                f"border-radius:0 4px 4px 0;background:rgba(255,255,255,0.03);cursor:{cursor}'"
                f"{hover_style}{onclick}"
                f" data-label=\"{safe_label}\" data-summary=\"{safe_summary}\" data-color=\"#58a6ff\">"
                f"<span style='font-weight:600;font-size:11px;color:#e6edf3'>{label}</span>"
                + (f"<div style='font-size:10px;color:#8b949e;margin-top:2px'>{preview}</div>" if preview else "")
                + "</div>"
            )

        feed_html = ""

        if ps_updates:
            feed_html += "<div style='font-size:11px;font-weight:700;color:#58a6ff;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.5px'>Phase Sequences</div>"
            for ps in ps_updates:
                ps_id   = ps.get("id", "")
                ps_name = ps.get("name", ps_id)
                safe_id   = ps_id.replace('"', "&quot;").replace("'", "&#39;")
                feed_html += (
                    f"<div style='margin-bottom:6px;padding:4px 6px;border-left:3px solid #58a6ff;"
                    f"border-radius:0 4px 4px 0;background:rgba(255,255,255,0.03);cursor:pointer'"
                    f" onmouseover=\"this.style.background='rgba(88,166,255,0.06)'\""
                    f" onmouseout=\"this.style.background='rgba(255,255,255,0.03)'\""
                    f" onclick=\"selectPS('{safe_id}')\">"
                    f"<span style='font-weight:600;font-size:11px;color:#e6edf3'>{ps_name}</span>"
                    f"<span style='font-size:10px;color:#58a6ff;margin-left:6px'>{ps_id}</span>"
                    f"</div>"
                )

        if conscious_updates:
            feed_html += "<div style='font-size:11px;font-weight:700;color:#f0c040;margin-top:8px;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.5px'>Conscious Neurons</div>"
            for p in conscious_updates:
                feed_html += news_item(p["name"], p.get("summary", ""))

        if spontaneous_updates:
            feed_html += "<div style='font-size:11px;font-weight:700;color:#8b949e;margin-top:8px;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.5px'>Spontaneous Neurons</div>"
            for p in spontaneous_updates:
                feed_html += news_item(p["name"], p.get("summary", ""))

        if feed_html:
            week = briefing.get("week_ending", "")
            week_label = f" — {week}" if week else ""
            html += (
                "\n  <hr style='border-color:#30363d;margin:10px 0'>"
                f"\n  <div style='font-size:12px;font-weight:700;margin-bottom:8px'>Weekly News{week_label}</div>"
                f"\n  <div style='max-height:380px;overflow-y:auto;padding-right:4px'>{feed_html}</div>"
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
# LAYOUT POSITION COMPUTATION
# ---------------------------------------------------------------------------

GOLDEN_ANGLE = 2.39996323  # radians (~137.5°)
CANVAS_SIZE  = 1000


def _name_jitter(name: str, canvas_size: int) -> tuple:
    """
    Deterministic (dx, dy) jitter derived from the node's name.
    Uses a simple polynomial hash — stable across runs, no imports needed.
    Magnitude is small enough to break circular symmetry without displacing
    nodes out of their cluster.
    """
    h = 0
    for c in name:
        h = (h * 31 + ord(c)) & 0xFFFFFF
    angle  = (h / 0x1000000) * 2 * math.pi
    radius = canvas_size * 0.03
    return radius * math.cos(angle), radius * math.sin(angle)


def _ps_anchor_circle(phase_sequences: list, radius: float) -> dict:
    """Return {ps_id: (x, y)} for PS anchors evenly spaced in a circle."""
    n = len(phase_sequences)
    anchors = {}
    for i, ps in enumerate(phase_sequences):
        angle = 2 * math.pi * i / n - math.pi / 2   # start at top
        anchors[ps["id"]] = (radius * math.cos(angle), radius * math.sin(angle))
    return anchors


def compute_neuron_positions(
    model_data: dict,
    coactivation_state: dict | None = None,
    canvas_size: int = CANVAS_SIZE,
) -> dict:
    """
    Compute deterministic (x, y) for every neuron.

    Strategy:
      - PS anchors placed evenly in a circle at 38% of canvas_size
      - Each neuron lands at the centroid of its PS anchors (boolean membership)
      - Neurons with no PS membership placed on an outer ring at 47% of canvas_size
      - Rank-based golden-angle scatter prevents stacking within clusters
      - Single-pass coactivation pull nudges co-firing pairs closer

    Returns {neuron_name: (x, y)}
    """
    cv  = model_data["canonical_vocabulary"]
    pss = cv.get("phase_sequences", [])
    cas = cv.get("cell_assemblies", [])

    anchors = _ps_anchor_circle(pss, canvas_size * 0.38)

    # Build neuron -> set of PS ids via CA membership
    neuron_ps: dict = {}
    for ca in cas:
        for neuron in ca.get("member_neurons", []):
            for ps_id in ca.get("ps_memberships", []):
                if ps_id in anchors:
                    neuron_ps.setdefault(neuron, set()).add(ps_id)

    people  = model_data["superorganism_list"]
    outer_r = canvas_size * 0.47

    positions: dict = {}

    # --- PS-assigned neurons: phyllotaxis spiral within each cluster ---
    # Group by centroid key (frozenset of PS ids) and sort by influence rank
    clusters: dict = {}
    for person in people:
        ps_set = neuron_ps.get(person["name"], set())
        if ps_set:
            clusters.setdefault(frozenset(ps_set), []).append(person)
    for key in clusters:
        clusters[key].sort(key=lambda p: p["rank"])

    for cluster_key, cluster_people in clusters.items():
        ps_set  = set(cluster_key)
        cx = sum(anchors[ps][0] for ps in ps_set) / len(ps_set)
        cy = sum(anchors[ps][1] for ps in ps_set) / len(ps_set)
        n  = len(cluster_people)
        base_r = canvas_size * 0.04 * math.sqrt(max(n, 1))
        for idx, person in enumerate(cluster_people):
            r     = base_r * math.sqrt(idx + 1)
            angle = idx * GOLDEN_ANGLE
            jx, jy = _name_jitter(person["name"], canvas_size)
            positions[person["name"]] = [
                cx + r * math.cos(angle) + jx,
                cy + r * math.sin(angle) + jy,
            ]

    # --- Unassigned neurons: golden spiral in outer region, sorted by rank ---
    unassigned = sorted(
        [p for p in people if not neuron_ps.get(p["name"])],
        key=lambda p: p["rank"],
    )
    outer_start_r = canvas_size * 0.44
    for idx, person in enumerate(unassigned):
        r     = outer_start_r + canvas_size * 0.03 * math.sqrt(idx)
        angle = idx * GOLDEN_ANGLE
        jx, jy = _name_jitter(person["name"], canvas_size)
        positions[person["name"]] = [
            r * math.cos(angle) + jx,
            r * math.sin(angle) + jy,
        ]

    # Single-pass coactivation pull
    if coactivation_state:
        threshold = coactivation_state.get("config", {}).get("edge_display_threshold", 0.15)
        for key, entry in coactivation_state.get("neuron_coactivation", {}).items():
            score = abs(entry.get("score", 0))
            if score < threshold:
                continue
            parts = key.split("|||")
            if len(parts) != 2:
                continue
            a, b = parts
            if a not in positions or b not in positions:
                continue
            ax, ay = positions[a]
            bx, by = positions[b]
            pull = score * 0.12
            positions[a] = [ax + (bx - ax) * pull, ay + (by - ay) * pull]
            positions[b] = [bx + (ax - bx) * pull, by + (ay - by) * pull]

    return positions


def compute_ca_positions(
    model_data: dict,
    coactivation_state: dict | None = None,
    canvas_size: int = CANVAS_SIZE,
) -> dict:
    """
    Compute deterministic (x, y) for every CA node.

    Same strategy as compute_neuron_positions but uses each CA's own
    ps_memberships field directly, and CA rank for golden-angle scatter.

    Returns {ca_id: (x, y)}
    """
    cv  = model_data["canonical_vocabulary"]
    pss = cv.get("phase_sequences", [])
    cas = cv.get("cell_assemblies", [])

    anchors = _ps_anchor_circle(pss, canvas_size * 0.38)

    outer_r = canvas_size * 0.47

    positions: dict = {}

    # --- PS-assigned CAs: phyllotaxis spiral within each cluster ---
    clusters: dict = {}
    for ca in cas:
        ps_set = frozenset(ps for ps in ca.get("ps_memberships", []) if ps in anchors)
        if ps_set:
            clusters.setdefault(ps_set, []).append(ca)
    for key in clusters:
        clusters[key].sort(key=lambda c: c.get("rank", 0))

    for cluster_key, cluster_cas in clusters.items():
        ps_set = set(cluster_key)
        cx = sum(anchors[ps][0] for ps in ps_set) / len(ps_set)
        cy = sum(anchors[ps][1] for ps in ps_set) / len(ps_set)
        n  = len(cluster_cas)
        base_r = canvas_size * 0.04 * math.sqrt(max(n, 1))
        for idx, ca in enumerate(cluster_cas):
            r     = base_r * math.sqrt(idx + 1)
            angle = idx * GOLDEN_ANGLE
            jx, jy = _name_jitter(ca["name"], canvas_size)
            positions[ca["id"]] = [
                cx + r * math.cos(angle) + jx,
                cy + r * math.sin(angle) + jy,
            ]

    # --- Unassigned CAs: golden spiral in outer region, sorted by rank ---
    unassigned_cas = sorted(
        [ca for ca in cas if not any(ps in anchors for ps in ca.get("ps_memberships", []))],
        key=lambda c: c.get("rank", 0),
    )
    outer_start_r = canvas_size * 0.44
    for idx, ca in enumerate(unassigned_cas):
        r     = outer_start_r + canvas_size * 0.03 * math.sqrt(idx)
        angle = idx * GOLDEN_ANGLE
        jx, jy = _name_jitter(ca["name"], canvas_size)
        positions[ca["id"]] = [
            r * math.cos(angle) + jx,
            r * math.sin(angle) + jy,
        ]

    # Single-pass CA coactivation pull
    if coactivation_state:
        threshold = coactivation_state.get("config", {}).get("edge_display_threshold", 0.15)
        for key, entry in coactivation_state.get("ca_coactivation", {}).items():
            score = abs(entry.get("score", 0))
            if score < threshold:
                continue
            parts = key.split("|||")
            if len(parts) != 2:
                continue
            a, b = parts
            if a not in positions or b not in positions:
                continue
            ax, ay = positions[a]
            bx, by = positions[b]
            pull = score * 0.12
            positions[a] = [ax + (bx - ax) * pull, ay + (by - ay) * pull]
            positions[b] = [bx + (ax - bx) * pull, by + (ay - by) * pull]

    return positions


# ---------------------------------------------------------------------------
# GLOBAL NETWORK BUILDER
# ---------------------------------------------------------------------------

def global_node_size(rank: int, n_total: int) -> int:
    """Scale node size by rank, compressed for large models."""
    return max(5, int(22 - (rank - 1) * 17 / max(n_total - 1, 1)))


def build_network(
    global_data: dict,
    hebbian_state: dict | None = None,
    coactivation_state: dict | None = None,
    positions: dict | None = None,
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

        pos = positions.get(person["name"], [0, 0]) if positions else [None, None]
        net.add_node(
            person["name"],
            label=person["name"],
            color={"background": color, "border": border,
                   "highlight": {"background": color, "border": "#ffffff"}},
            size=size,
            font={"size": 13, "color": FONT_COLOR, "face": "sans-serif"},
            borderWidth=2,
            x=pos[0],
            y=pos[1],
        )

    # --- Edges from coactivation state (no fallback — no state = no edges) ---
    name_set = {p["name"] for p in people}
    if coactivation_state:
        threshold = coactivation_state.get("config", {}).get("edge_display_threshold", 0.15)
        for key, entry in coactivation_state.get("neuron_coactivation", {}).items():
            if abs(entry["score"]) < threshold:
                continue
            parts = key.split("|||")
            if len(parts) != 2:
                continue
            name_a, name_b = parts
            if name_a not in name_set or name_b not in name_set:
                continue
            score  = entry["score"]
            ecolor = EXCITATORY_COLOR if score > 0 else INHIBITORY_COLOR
            last_ps = ", ".join(entry.get("last_ps", []))
            label_str = entry["label"].capitalize()
            label_color = EXCITATORY_COLOR if score > 0 else INHIBITORY_COLOR
            edge_key = f"{name_a}|||{name_b}"
            edge_tooltips[edge_key] = (
                f"<div style='font-family:sans-serif;font-size:13px;color:#e6edf3;max-width:280px'>"
                f"<div style='font-weight:600;margin-bottom:6px'>"
                f"<span style='color:{label_color}'>{label_str}</span></div>"
                f"<div style='color:#8b949e;font-size:11px'>Score: {score:.3f}</div>"
                + (f"<div style='color:#8b949e;font-size:11px'>Last PS: {last_ps}</div>" if last_ps else "")
                + f"<div style='color:#8b949e;font-size:11px'>Observations: {entry.get('observations', 0)}</div>"
                f"</div>"
            )
            net.add_edge(
                name_a, name_b,
                value=abs(score),
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
      "physics": { "enabled": false },
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



def us_node_size(rank: int, n_total: int) -> int:
    """Scale node size for US model."""
    return max(5, int(22 - (rank - 1) * 17 / max(n_total - 1, 1)))


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


def load_latest_global_briefing(script_dir: str) -> dict | None:
    briefings_dir = os.path.join(script_dir, "briefings")
    if not os.path.isdir(briefings_dir):
        return None
    files = sorted(
        (f for f in os.listdir(briefings_dir)
         if f.startswith("global_weekly_briefing_") and f.endswith(".json")
         and "_raw_" not in f),
        reverse=True,
    )
    if not files:
        return None
    path = os.path.join(briefings_dir, files[0])
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def load_coactivation_state(script_dir: str, scope: str = "us") -> dict | None:
    filename = "us_coactivation_state.json" if scope == "us" else "global_coactivation_state.json"
    path = os.path.join(script_dir, "state", "coactivation", filename)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_ps_ca_membership_edges(model_data: dict) -> dict:
    """
    Build {ps_id: [{from, to, color, value}, ...]} for the assembly view.
    Edges connect CA pairs that share the same PS.
    """
    raw_asm = model_data.get("canonical_vocabulary", {}).get("cell_assemblies", [])

    ps_to_cas: dict = {}
    for ca in raw_asm:
        if isinstance(ca, str):
            continue
        ca_id = ca.get("id") or ca.get("name", "")
        for ps_id in ca.get("ps_memberships", []):
            ps_to_cas.setdefault(ps_id, set()).add(ca_id)

    result = {}
    for ps_id, ca_ids in ps_to_cas.items():
        ca_list = sorted(ca_ids)
        edges = [
            {"from": a, "to": b,
             "color": {"color": "#ffffff", "opacity": 0.4},
             "value": 1}
            for i, a in enumerate(ca_list)
            for b in ca_list[i + 1:]
        ]
        if edges:
            result[ps_id] = edges
    return result


def build_ps_membership_edges(model_data: dict) -> dict:
    """
    Build {ps_id: [{from, to, color, value}, ...]} for the PS co-membership selector.
    Edges are structural (from model CA membership), not from coactivation scores.
    """
    people   = model_data["superorganism_list"]
    raw_asm  = model_data.get("canonical_vocabulary", {}).get("cell_assemblies", [])

    ps_to_cas: dict = {}
    for ca in raw_asm:
        if isinstance(ca, str):
            continue
        ca_id = ca.get("id") or ca.get("name", "")
        for ps_id in ca.get("ps_memberships", []):
            ps_to_cas.setdefault(ps_id, set()).add(ca_id)

    ca_to_neurons: dict = {}
    for person in people:
        for ca in person.get("superorganism", {}).get("cell_assemblies", []):
            ca_id = ca.get("id") or ca.get("name", "")
            if ca_id:
                ca_to_neurons.setdefault(ca_id, set()).add(person["name"])

    result = {}
    for ps_id, ca_ids in ps_to_cas.items():
        neurons_in_ps = set()
        for ca_id in ca_ids:
            neurons_in_ps |= ca_to_neurons.get(ca_id, set())
        neuron_list = sorted(neurons_in_ps)
        edges = [
            {"from": a, "to": b,
             "color": {"color": "#ffffff", "opacity": 0.4},
             "value": 1}
            for i, a in enumerate(neuron_list)
            for b in neuron_list[i + 1:]
        ]
        if edges:
            result[ps_id] = edges
    return result


# ---------------------------------------------------------------------------
# LEGEND BUILDERS
# ---------------------------------------------------------------------------

def build_us_legend_html(briefing: dict | None = None) -> str:
    html = """<div id="us-legend" style="
    display:none;
    position:fixed; top:20px; right:20px; z-index:9999;
    background:rgba(13,17,23,0.92); border:1px solid #30363d;
    border-radius:8px; padding:14px 18px; font-family:sans-serif;
    font-size:13px; color:#e6edf3; min-width:260px; max-width:320px; line-height:1.8">
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

    if briefing:
        meta = briefing.get("_metadata", {})
        conscious_set = set(meta.get("neurons_conscious", []))
        spontaneous_set = set(meta.get("neurons_spontaneous", []))

        # Phase Sequences
        ps_updates = briefing.get("phase_sequence_updates", [])
        # Cell Assemblies — active only
        ca_updates = [a for a in briefing.get("assembly_updates", []) if a.get("signal", "active") == "active"]
        # Neurons split by type — active only
        person_updates = briefing.get("person_updates", [])
        conscious_updates = [
            p for p in person_updates
            if p["name"] in conscious_set and p.get("signal", "active") == "active"
        ]
        spontaneous_updates = [
            p for p in person_updates
            if p["name"] in spontaneous_set and p.get("signal", "active") == "active"
        ]

        def news_item(label: str, summary: str) -> str:
            safe_label   = label.replace('"', "&quot;").replace("'", "&#39;") if label else ""
            safe_summary = summary.replace('"', "&quot;").replace("'", "&#39;") if summary else ""
            preview  = summary[:120] + ("…" if len(summary) > 120 else "") if summary else ""
            clickable = bool(summary)
            cursor   = "pointer" if clickable else "default"
            onclick  = ' onclick="openNewsModal(this)"' if clickable else ""
            hover_style = (
                " onmouseover=\"this.style.background='rgba(88,166,255,0.06)'\" "
                "onmouseout=\"this.style.background='rgba(255,255,255,0.03)'\""
                if clickable else ""
            )
            return (
                f"<div style='margin-bottom:6px;padding:4px 6px;border-left:3px solid #58a6ff;"
                f"border-radius:0 4px 4px 0;background:rgba(255,255,255,0.03);cursor:{cursor}'"
                f"{hover_style}{onclick}"
                f" data-label=\"{safe_label}\" data-summary=\"{safe_summary}\" data-color=\"#58a6ff\">"
                f"<span style='font-weight:600;font-size:11px;color:#e6edf3'>{label}</span>"
                + (f"<div style='font-size:10px;color:#8b949e;margin-top:2px'>{preview}</div>" if preview else "")
                + "</div>"
            )

        feed_html = ""

        if ps_updates:
            feed_html += "<div style='font-size:11px;font-weight:700;color:#58a6ff;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.5px'>Phase Sequences</div>"
            for ps in ps_updates:
                ps_id  = ps.get("id", "")
                ps_name = ps.get("name", ps_id)
                safe_name = ps_name.replace('"', "&quot;").replace("'", "&#39;")
                safe_id   = ps_id.replace('"', "&quot;").replace("'", "&#39;")
                feed_html += (
                    f"<div style='margin-bottom:6px;padding:4px 6px;border-left:3px solid #58a6ff;"
                    f"border-radius:0 4px 4px 0;background:rgba(255,255,255,0.03);cursor:pointer'"
                    f" onmouseover=\"this.style.background='rgba(88,166,255,0.06)'\""
                    f" onmouseout=\"this.style.background='rgba(255,255,255,0.03)'\""
                    f" onclick=\"selectPS('{safe_id}')\">"
                    f"<span style='font-weight:600;font-size:11px;color:#e6edf3'>{ps_name}</span>"
                    f"<span style='font-size:10px;color:#58a6ff;margin-left:6px'>{ps_id}</span>"
                    f"</div>"
                )

        if conscious_updates:
            feed_html += "<div style='font-size:11px;font-weight:700;color:#f0c040;margin-top:8px;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.5px'>Conscious Neurons</div>"
            for p in conscious_updates:
                feed_html += news_item(p["name"], p.get("summary", ""))

        if spontaneous_updates:
            feed_html += "<div style='font-size:11px;font-weight:700;color:#8b949e;margin-top:8px;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.5px'>Spontaneous Neurons</div>"
            for p in spontaneous_updates:
                feed_html += news_item(p["name"], p.get("summary", ""))

        if feed_html:
            week = briefing.get("week_ending", "")
            week_label = f" — {week}" if week else ""
            html += (
                "\n  <hr style='border-color:#30363d;margin:10px 0'>"
                f"\n  <div style='font-size:12px;font-weight:700;margin-bottom:8px'>Weekly News{week_label}</div>"
                f"\n  <div style='max-height:380px;overflow-y:auto;padding-right:4px'>{feed_html}</div>"
            )

    html += "\n</div>\n"
    return html


def build_us_asm_legend_html(briefing: dict | None = None) -> str:
    """Assembly-view sidebar: static legend + PS and CA news feed."""
    html = """<div id="us-asm-legend" style="
    display:none;
    position:fixed; top:20px; right:20px; z-index:9999;
    background:rgba(13,17,23,0.92); border:1px solid #30363d;
    border-radius:8px; padding:14px 18px; font-family:sans-serif;
    font-size:13px; color:#e6edf3; min-width:260px; max-width:320px; line-height:1.8">
  <div style="font-size:14px;font-weight:700;margin-bottom:8px">US &middot; Assemblies</div>
  <div><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#4a90d9;margin-right:8px;vertical-align:middle"></span>All nodes: West (US)</div>
  <hr style="border-color:#30363d;margin:10px 0">
  <div style="color:#8b949e;font-size:11px">
    Node size = member count<br>
    Edge = shared phase sequences<br>
    Hover for members &amp; news
  </div>"""

    if briefing:
        def news_item(label: str, summary: str) -> str:
            safe_label   = label.replace('"', "&quot;").replace("'", "&#39;") if label else ""
            safe_summary = summary.replace('"', "&quot;").replace("'", "&#39;") if summary else ""
            preview  = summary[:120] + ("…" if len(summary) > 120 else "") if summary else ""
            clickable = bool(summary)
            cursor   = "pointer" if clickable else "default"
            onclick  = ' onclick="openNewsModal(this)"' if clickable else ""
            hover_style = (
                " onmouseover=\"this.style.background='rgba(88,166,255,0.06)'\" "
                "onmouseout=\"this.style.background='rgba(255,255,255,0.03)'\""
                if clickable else ""
            )
            return (
                f"<div style='margin-bottom:6px;padding:4px 6px;border-left:3px solid #58a6ff;"
                f"border-radius:0 4px 4px 0;background:rgba(255,255,255,0.03);cursor:{cursor}'"
                f"{hover_style}{onclick}"
                f" data-label=\"{safe_label}\" data-summary=\"{safe_summary}\" data-color=\"#58a6ff\">"
                f"<span style='font-weight:600;font-size:11px;color:#e6edf3'>{label}</span>"
                + (f"<div style='font-size:10px;color:#8b949e;margin-top:2px'>{preview}</div>" if preview else "")
                + "</div>"
            )

        ps_updates = briefing.get("phase_sequence_updates", [])
        ca_updates = [a for a in briefing.get("assembly_updates", []) if a.get("signal", "active") == "active"]
        feed_html  = ""

        if ps_updates:
            feed_html += "<div style='font-size:11px;font-weight:700;color:#58a6ff;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.5px'>Phase Sequences</div>"
            for ps in ps_updates:
                ps_id  = ps.get("id", "")
                ps_name = ps.get("name", ps_id)
                safe_id   = ps_id.replace('"', "&quot;").replace("'", "&#39;")
                feed_html += (
                    f"<div style='margin-bottom:6px;padding:4px 6px;border-left:3px solid #58a6ff;"
                    f"border-radius:0 4px 4px 0;background:rgba(255,255,255,0.03);cursor:pointer'"
                    f" onmouseover=\"this.style.background='rgba(88,166,255,0.06)'\""
                    f" onmouseout=\"this.style.background='rgba(255,255,255,0.03)'\""
                    f" onclick=\"selectPS('{safe_id}')\">"
                    f"<span style='font-weight:600;font-size:11px;color:#e6edf3'>{ps_name}</span>"
                    f"<span style='font-size:10px;color:#58a6ff;margin-left:6px'>{ps_id}</span>"
                    f"</div>"
                )

        if ca_updates:
            feed_html += "<div style='font-size:11px;font-weight:700;color:#a78bfa;margin-top:8px;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.5px'>Cell Assemblies</div>"
            for ca in ca_updates:
                feed_html += news_item(ca.get("name", ca.get("id", "")), ca.get("summary", ""))

        if feed_html:
            week = briefing.get("week_ending", "")
            week_label = f" — {week}" if week else ""
            html += (
                "\n  <hr style='border-color:#30363d;margin:10px 0'>"
                f"\n  <div style='font-size:12px;font-weight:700;margin-bottom:8px'>Weekly News{week_label}</div>"
                f"\n  <div style='max-height:380px;overflow-y:auto;padding-right:4px'>{feed_html}</div>"
            )

    html += "\n</div>\n"
    return html


NEWS_MODAL_HTML = """
<div id="news-modal" onclick="if(event.target===this)closeNewsModal()" style="
    display:none; position:fixed; inset:0; z-index:19999;
    background:rgba(0,0,0,0.6); align-items:center; justify-content:center">
  <div style="
      background:#161b22; border:1px solid #30363d; border-radius:10px;
      padding:20px 24px; max-width:480px; width:90%; font-family:sans-serif;
      color:#e6edf3; position:relative; box-shadow:0 8px 32px rgba(0,0,0,0.6)">
    <div id="news-modal-border" style="
        position:absolute; left:0; top:0; bottom:0; width:4px; border-radius:10px 0 0 10px"></div>
    <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px">
      <div id="news-modal-label" style="font-size:14px;font-weight:700;line-height:1.4;flex:1"></div>
      <button onclick="closeNewsModal()" style="
          background:none;border:none;color:#8b949e;font-size:18px;cursor:pointer;
          padding:0;line-height:1;flex-shrink:0" title="Close">&times;</button>
    </div>
    <div id="news-modal-body" style="
        font-size:13px;color:#c9d1d9;margin-top:12px;line-height:1.7;
        max-height:60vh;overflow-y:auto"></div>
  </div>
</div>
<script>
function openNewsModal(el) {
  var label   = el.getAttribute('data-label')   || '';
  var summary = el.getAttribute('data-summary') || '';
  var color   = el.getAttribute('data-color')   || '#8b949e';
  document.getElementById('news-modal-label').textContent  = label;
  document.getElementById('news-modal-body').textContent   = summary;
  document.getElementById('news-modal-border').style.background = color;
  var modal = document.getElementById('news-modal');
  modal.style.display = 'flex';
}
function closeNewsModal() {
  document.getElementById('news-modal').style.display = 'none';
}
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') closeNewsModal();
});
</script>
"""


# ---------------------------------------------------------------------------
# COMBINED-VIZ SPECIFIC HTML OVERLAYS
# ---------------------------------------------------------------------------

COMBINED_TITLE_HTML = """
<div id="graph-title-block" style="position:fixed;top:20px;left:20px;z-index:9999;font-family:sans-serif;color:#e6edf3">
  <div id="graph-title" style="font-size:20px;font-weight:bold;letter-spacing:0.5px">Human Superorganism</div>
  <div id="graph-subtitle" style="font-size:12px;color:#8b949e;margin-top:2px">Prime Movers</div>
</div>
"""

def build_global_asm_legend_html(briefing: dict | None = None) -> str:
    """Global assembly-view sidebar with optional weekly news feed."""
    html = """<div id="global-asm-legend" style="
    display:none;
    position:fixed; top:20px; right:20px; z-index:9999;
    background:rgba(13,17,23,0.92); border:1px solid #30363d;
    border-radius:8px; padding:14px 18px; font-family:sans-serif;
    font-size:13px; color:#e6edf3; min-width:200px; max-width:320px; line-height:1.8">
  <div style="font-size:14px;font-weight:700;margin-bottom:8px">Global &middot; Assemblies</div>
  <div><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#4a90d9;margin-right:8px;vertical-align:middle"></span>West-dominated</div>
  <div><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#e05c5c;margin-right:8px;vertical-align:middle"></span>East-dominated</div>
  <div><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#a78bfa;margin-right:8px;vertical-align:middle"></span>Bridge</div>
  <hr style="border-color:#30363d;margin:10px 0">
  <div style="color:#8b949e;font-size:11px">
    Node size = member count<br>
    Edge = shared phase sequences<br>
    Hover for members &amp; news
  </div>"""

    if briefing:
        def news_item(label: str, summary: str) -> str:
            safe_label   = label.replace('"', "&quot;").replace("'", "&#39;") if label else ""
            safe_summary = summary.replace('"', "&quot;").replace("'", "&#39;") if summary else ""
            preview  = summary[:120] + ("…" if len(summary) > 120 else "") if summary else ""
            clickable = bool(summary)
            cursor   = "pointer" if clickable else "default"
            onclick  = ' onclick="openNewsModal(this)"' if clickable else ""
            hover_style = (
                " onmouseover=\"this.style.background='rgba(88,166,255,0.06)'\" "
                "onmouseout=\"this.style.background='rgba(255,255,255,0.03)'\""
                if clickable else ""
            )
            return (
                f"<div style='margin-bottom:6px;padding:4px 6px;border-left:3px solid #58a6ff;"
                f"border-radius:0 4px 4px 0;background:rgba(255,255,255,0.03);cursor:{cursor}'"
                f"{hover_style}{onclick}"
                f" data-label=\"{safe_label}\" data-summary=\"{safe_summary}\" data-color=\"#58a6ff\">"
                f"<span style='font-weight:600;font-size:11px;color:#e6edf3'>{label}</span>"
                + (f"<div style='font-size:10px;color:#8b949e;margin-top:2px'>{preview}</div>" if preview else "")
                + "</div>"
            )

        ps_updates = briefing.get("phase_sequence_updates", [])
        ca_updates = [a for a in briefing.get("assembly_updates", []) if a.get("signal", "active") == "active"]
        feed_html  = ""

        if ps_updates:
            feed_html += "<div style='font-size:11px;font-weight:700;color:#58a6ff;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.5px'>Phase Sequences</div>"
            for ps in ps_updates:
                ps_id   = ps.get("id", "")
                ps_name = ps.get("name", ps_id)
                safe_id   = ps_id.replace('"', "&quot;").replace("'", "&#39;")
                feed_html += (
                    f"<div style='margin-bottom:6px;padding:4px 6px;border-left:3px solid #58a6ff;"
                    f"border-radius:0 4px 4px 0;background:rgba(255,255,255,0.03);cursor:pointer'"
                    f" onmouseover=\"this.style.background='rgba(88,166,255,0.06)'\""
                    f" onmouseout=\"this.style.background='rgba(255,255,255,0.03)'\""
                    f" onclick=\"selectPS('{safe_id}')\">"
                    f"<span style='font-weight:600;font-size:11px;color:#e6edf3'>{ps_name}</span>"
                    f"<span style='font-size:10px;color:#58a6ff;margin-left:6px'>{ps_id}</span>"
                    f"</div>"
                )

        if ca_updates:
            feed_html += "<div style='font-size:11px;font-weight:700;color:#a78bfa;margin-top:8px;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.5px'>Cell Assemblies</div>"
            for ca in ca_updates:
                feed_html += news_item(ca.get("name", ca.get("id", "")), ca.get("summary", ""))

        if feed_html:
            week = briefing.get("week_ending", "")
            week_label = f" — {week}" if week else ""
            html += (
                "\n  <hr style='border-color:#30363d;margin:10px 0'>"
                f"\n  <div style='font-size:12px;font-weight:700;margin-bottom:8px'>Weekly News{week_label}</div>"
                f"\n  <div style='max-height:380px;overflow-y:auto;padding-right:4px'>{feed_html}</div>"
            )

    html += "\n</div>\n"
    return html


# ---------------------------------------------------------------------------
# PS DETAIL PANEL
# ---------------------------------------------------------------------------

PS_DETAIL_PANEL_HTML = """<div id="ps-detail-panel" style="
    display:none;
    position:fixed; bottom:20px; right:20px; z-index:10000;
    background:rgba(13,17,23,0.97); border:1px solid #58a6ff;
    border-radius:8px; padding:14px 18px; font-family:sans-serif;
    font-size:13px; color:#e6edf3; min-width:280px; max-width:420px;
    max-height:65vh; overflow-y:auto;
    box-shadow:0 4px 20px rgba(88,166,255,0.25)">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
    <span style="font-size:11px;color:#58a6ff;font-weight:600;letter-spacing:0.5px">PHASE SEQUENCE</span>
    <button onclick="hidePSPanel()" style="background:none;border:none;color:#8b949e;
      cursor:pointer;font-size:18px;padding:0;line-height:1">&times;</button>
  </div>
  <div id="ps-detail-content"></div>
</div>
"""


def build_ps_panel_data(briefing: dict | None) -> dict:
    """Build {ps_id: {name, summary}} from briefing phase_sequence_updates."""
    if not briefing:
        return {}
    result = {}
    for upd in briefing.get("phase_sequence_updates", []):
        result[upd["id"]] = {
            "name":    upd.get("name", upd["id"]),
            "summary": upd.get("summary", ""),
        }
    return result


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

    search_row = (
        "<div style='display:flex;margin-top:6px'>"
        "<div style='position:relative'>"
        "<input id='node-search' type='text' placeholder='Search neurons...' "
        "oninput='performSearch(this.value)' "
        "style='background:#161b22;border:1px solid #30363d;border-radius:6px;"
        "padding:6px 26px 6px 10px;font-size:12px;color:#e6edf3;outline:none;"
        "width:210px;box-shadow:0 2px 4px rgba(0,0,0,0.4);'>"
        "<span id='search-clear' onclick='clearSearch()' style='"
        "display:none;position:absolute;right:7px;top:50%;"
        "transform:translateY(-50%);color:#8b949e;cursor:pointer;"
        "font-size:16px;line-height:1'>&times;</span>"
        "</div>"
        "</div>"
    )

    return (
        "<div id='view-controls' style='position:fixed;top:64px;left:50%;"
        "transform:translateX(-50%);z-index:9999;font-family:sans-serif;"
        "display:flex;flex-direction:column;align-items:center'>"
        f"{scope_row}{type_row}{search_row}"
        "</div>"
    )


# ---------------------------------------------------------------------------
# US NETWORK BUILDER
# ---------------------------------------------------------------------------

def build_us_network(
    us_data: dict,
    briefing: dict | None = None,
    hebbian_state: dict | None = None,
    coactivation_state: dict | None = None,
    positions: dict | None = None,
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

    person_summaries  = {}
    person_top_stories: dict = {}
    briefing_week    = ""
    if briefing:
        briefing_week = briefing.get("week_ending", "")
        for pu in briefing.get("person_updates", []):
            if pu.get("signal", "active") == "active":
                person_summaries[pu["name"]] = pu.get("summary", "")
        for story in briefing.get("top_stories", []):
            for name in story.get("persons", []):
                person_top_stories.setdefault(name, []).append(story)

    # --- Nodes ---
    for person in people:
        so   = person["superorganism"]
        size = us_node_size(person["rank"], n_total)
        ps_rows = ""
        for ps in so.get("phase_sequences", []):
            ps_rows += (
                f"<div style='margin-top:6px'>"
                f"<span style='color:#58a6ff;font-weight:600;font-size:11px'>{ps['id']}</span>"
                f"&nbsp;<span style='color:#e6edf3'>{ps['name']}</span><br>"
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

        weekly_html    = ""
        weekly_summary = person_summaries.get(person["name"], "")
        if weekly_summary:
            week_label  = f"This Week ({briefing_week})" if briefing_week else "This Week"
            weekly_html = (
                f"<hr style='border-color:#30363d;margin:10px 0'>"
                f"<div style='font-size:12px;font-weight:600;color:#e6edf3;"
                f"margin-bottom:4px'>{week_label}</div>"
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

        pos = positions.get(person["name"], [0, 0]) if positions else [None, None]
        net.add_node(
            person["name"],
            label=person["name"],
            color={"background": color, "border": border,
                   "highlight": {"background": color, "border": "#ffffff"}},
            size=size,
            font={"size": 13, "color": FONT_COLOR, "face": "sans-serif"},
            borderWidth=2,
            x=pos[0],
            y=pos[1],
        )

    # --- Edges from coactivation state ---
    name_set   = {p["name"] for p in people}
    threshold  = 0.0
    if coactivation_state:
        threshold = coactivation_state.get("config", {}).get("edge_display_threshold", 0.15)
        for key, entry in coactivation_state.get("neuron_coactivation", {}).items():
            if abs(entry["score"]) < threshold:
                continue
            parts = key.split("|||")
            if len(parts) != 2:
                continue
            name_a, name_b = parts
            if name_a not in name_set or name_b not in name_set:
                continue
            score  = entry["score"]
            ecolor = EXCITATORY_COLOR if score > 0 else INHIBITORY_COLOR
            last_ps = ", ".join(entry.get("last_ps", []))
            label_str = entry["label"].capitalize()
            label_color = EXCITATORY_COLOR if score > 0 else INHIBITORY_COLOR

            edge_key = f"{name_a}|||{name_b}"
            edge_tooltips[edge_key] = (
                f"<div style='font-family:sans-serif;font-size:13px;color:#e6edf3;max-width:280px'>"
                f"<div style='font-weight:600;margin-bottom:6px'>"
                f"<span style='color:{label_color}'>{label_str}</span></div>"
                f"<div style='color:#8b949e;font-size:11px'>Score: {score:.3f}</div>"
                + (f"<div style='color:#8b949e;font-size:11px'>Last PS: {last_ps}</div>" if last_ps else "")
                + f"<div style='color:#8b949e;font-size:11px'>Observations: {entry.get('observations', 0)}</div>"
                f"</div>"
            )
            net.add_edge(
                name_a, name_b,
                value=abs(score),
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
      "physics": { "enabled": false },
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
    coactivation_state: dict | None = None,
    positions: dict | None = None,
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
        size     = max(5, int(6 + 16 * n_members / max(max_members, 1)))

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

        pos = positions.get(ca["id"], [0, 0]) if positions else [None, None]
        net.add_node(
            ca["id"],
            label=ca["name"],
            color={"background": color, "border": border,
                   "highlight": {"background": color, "border": "#ffffff"}},
            size=size,
            font={"size": 12, "color": FONT_COLOR, "face": "sans-serif"},
            borderWidth=2,
            x=pos[0],
            y=pos[1],
        )

    # Edges: coactivation scores only (no fallback — no state = no edges)
    ca_id_set = {ca["id"] for ca in assemblies}
    if coactivation_state:
        threshold = coactivation_state.get("config", {}).get("edge_display_threshold", 0.15)
        for key, entry in coactivation_state.get("ca_coactivation", {}).items():
            if abs(entry["score"]) < threshold:
                continue
            parts = key.split("|||")
            if len(parts) != 2:
                continue
            ca_a_id, ca_b_id = parts
            if ca_a_id not in ca_id_set or ca_b_id not in ca_id_set:
                continue
            score  = entry["score"]
            ecolor = EXCITATORY_COLOR if score > 0 else INHIBITORY_COLOR
            last_ps = ", ".join(entry.get("last_ps", []))
            label_str = entry["label"].capitalize()
            label_color = EXCITATORY_COLOR if score > 0 else INHIBITORY_COLOR
            edge_key = f"{ca_a_id}|||{ca_b_id}"
            edge_tooltips[edge_key] = (
                f"<div style='font-family:sans-serif;font-size:13px;color:#e6edf3;max-width:280px'>"
                f"<div style='font-weight:600;margin-bottom:6px'>"
                f"<span style='color:{label_color}'>{label_str}</span></div>"
                f"<div style='color:#8b949e;font-size:11px'>Score: {score:.3f}</div>"
                + (f"<div style='color:#8b949e;font-size:11px'>Last PS: {last_ps}</div>" if last_ps else "")
                + f"<div style='color:#8b949e;font-size:11px'>Observations: {entry.get('observations', 0)}</div>"
                f"</div>"
            )
            net.add_edge(
                ca_a_id, ca_b_id,
                value=abs(score),
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
        "scaling": { "min": 1, "max": 6 }
      },
      "physics": { "enabled": false },
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
# PS PANEL JS BUILDER
# ---------------------------------------------------------------------------

def _build_ps_panel_js(ps_panel_data: dict) -> str:
    """Return a <script> block that defines psPanelData, showPSPanel, hidePSPanel."""
    data_json = json.dumps(ps_panel_data, ensure_ascii=False)
    return (
        "<script type=\"text/javascript\">\n"
        f"var psPanelData = {data_json};\n"
        "window.showPSPanel = function(psId) {\n"
        "  var ps = psPanelData[psId];\n"
        "  if (!ps) return;\n"
        "  var html = \"<div style='font-size:15px;font-weight:700;margin-bottom:6px'>\" + ps.name + \"<span style='color:#58a6ff;font-size:10px;margin-left:8px'>\" + psId + \"</span></div>\"\n"
        "    + \"<hr style='border-color:#30363d;margin:10px 0'>\"\n"
        "    + \"<div style='color:#8b949e;font-size:12px;line-height:1.7'>\" + (ps.summary || 'No synthesis available.') + \"</div>\";\n"
        "  document.getElementById('ps-detail-content').innerHTML = html;\n"
        "  document.getElementById('ps-detail-panel').style.display = 'block';\n"
        "};\n"
        "window.hidePSPanel = function() {\n"
        "  document.getElementById('ps-detail-panel').style.display = 'none';\n"
        "};\n"
        "</script>\n"
    )


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
    # PS detail panel data
    ps_panel_data=None,
    # PS co-membership edges for PS selector
    ps_membership_edges=None,
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
    // Remove pyvis loading-bar handlers so the bar never appears
    network.off('stabilizationProgress');
    network.off('stabilizationIterationsDone');
    var lb = document.getElementById('loadingBar');
    if (lb) lb.style.display = 'none';
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

    ps_selector_js = (
        "<script type=\"text/javascript\">\n"
        f"var psMembershipEdges = {json.dumps(ps_membership_edges or {}, ensure_ascii=False)};\n"
        "var selectedPS = null;\n"
        "window.selectPS = function(psId) {\n"
        "  var viewKey = currentScope + '_' + currentViewType;\n"
        "  var datasets = {\n"
        "    'global_neuron':   {nodes: globalNeuronNodes, edges: globalNeuronEdges},\n"
        "    'us_neuron':       {nodes: usNeuronNodes,     edges: usNeuronEdges},\n"
        "    'global_assembly': {nodes: globalAsmNodes,    edges: globalAsmEdges},\n"
        "    'us_assembly':     {nodes: usAsmNodes,        edges: usAsmEdges},\n"
        "  };\n"
        "  var currentDs = datasets[viewKey];\n"
        "  if (!currentDs) return;\n"
        "  if (selectedPS === psId) {\n"
        "    selectedPS = null;\n"
        "    network.setData(currentDs);\n"
        "    window.updateTooltipMap(currentScope, currentViewType);\n"
        "    hidePSPanel();\n"
        "  } else {\n"
        "    selectedPS = psId;\n"
        "    var edgeMap = psMembershipEdges[viewKey] || {};\n"
        "    var rawEdges = edgeMap[psId] || [];\n"
        "    var tempEdges = new vis.DataSet(rawEdges.map(function(e, i) {\n"
        "      return {id: 'ps_' + i, from: e.from, to: e.to, color: e.color, value: e.value};\n"
        "    }));\n"
        "    network.setData({nodes: currentDs.nodes, edges: tempEdges});\n"
        "    showPSPanel(psId);\n"
        "  }\n"
        "};\n"
        "</script>\n"
    )

    switch_view_js = """<script type="text/javascript">
window.switchView = function(scope, viewType) {
  selectedPS = null;
  hidePSPanel();
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
  // Clear search and update placeholder for new view
  if (typeof clearSearch === 'function') clearSearch();
  var searchEl = document.getElementById('node-search');
  if (searchEl) searchEl.placeholder = viewType === 'assembly' ? 'Search assemblies...' : 'Search neurons...';
};
</script>
"""

    controls_html = make_controls_html(n_global, n_us, n_global_asm, n_us_asm, has_global)

    search_js = """<script type="text/javascript">
var searchQuery = '';
window.performSearch = function(query) {
  searchQuery = query.trim().toLowerCase();
  var clearBtn = document.getElementById('search-clear');
  if (clearBtn) clearBtn.style.display = searchQuery ? 'inline' : 'none';
  var nodeMap = {
    'global_neuron':   globalNeuronNodes,
    'us_neuron':       usNeuronNodes,
    'global_assembly': globalAsmNodes,
    'us_assembly':     usAsmNodes,
  };
  var currentNodes = nodeMap[currentScope + '_' + currentViewType];
  if (!currentNodes) return;
  var updates = [];
  currentNodes.forEach(function(node) {
    var label = (node.label || String(node.id) || '').toLowerCase();
    if (!searchQuery || label.indexOf(searchQuery) !== -1) {
      var orig = nodeColors[node.id];
      if (orig) updates.push({id: node.id, color: orig});
    } else {
      updates.push({id: node.id, color: {
        background: '#2d333b', border: '#3d444d',
        highlight: {background: '#2d333b', border: '#3d444d'}
      }});
    }
  });
  currentNodes.update(updates);
};
window.clearSearch = function() {
  var inp = document.getElementById('node-search');
  if (inp) { inp.value = ''; performSearch(''); }
};
</script>
"""

    body_injection = (
        TOOLTIP_DIV
        + controls_html
        + (us_legend_html         or "")
        + (global_legend_html     or build_global_legend_html())
        + (global_asm_legend_html or build_global_asm_legend_html())
        + (us_asm_legend_html     or build_us_asm_legend_html())
        + COMBINED_TITLE_HTML
        + PS_DETAIL_PANEL_HTML
        + NEWS_MODAL_HTML
        + tooltip_dicts_js
        + tooltip_listener_js
        + switch_view_js
        + _build_ps_panel_js(ps_panel_data or {})
        + ps_selector_js
        + search_js
    )
    html = html.replace("</body>", body_injection + "</body>", 1)

    nav_html = (
        '<style>'
        'nav.pmt-nav{position:fixed;top:0;left:0;right:0;z-index:99999;'
        'height:52px;display:flex;align-items:center;justify-content:space-between;'
        'padding:0 2rem;background:rgba(13,17,23,0.92);backdrop-filter:blur(8px);'
        'border-bottom:1px solid #30363d;}'
        'nav.pmt-nav a{font-family:Inter,system-ui,sans-serif;font-size:0.82rem;'
        'font-weight:500;text-decoration:none;color:#8b949e;transition:color 0.15s;}'
        'nav.pmt-nav a:hover{color:#e6edf3;}'
        'nav.pmt-nav .brand{font-size:0.85rem;font-weight:600;letter-spacing:0.08em;'
        'text-transform:uppercase;color:#e6edf3;}'
        'nav.pmt-nav .nav-links{display:flex;gap:1.75rem;list-style:none;}'
        'nav.pmt-nav .nav-links a.active{color:#58a6ff;}'
        '#mynetwork{margin-top:52px;height:calc(100vh - 52px)!important;}'
        '</style>'
        '<nav class="pmt-nav">'
        '<a class="brand" href="/explainer.html">Prime Mover Tracker</a>'
        '<ul class="nav-links">'
        '<li><a href="/explainer.html">Explainer</a></li>'
        '<li><a href="/combined_viz.html" class="active">Visualization</a></li>'
        '</ul>'
        '</nav>'
    )
    html = html.replace("<body>", "<body>" + nav_html, 1)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Combined Superorganism Visualizer")
    parser.add_argument("--no-serve", action="store_true",
                        help="Write the HTML file and exit without starting the local server")
    args = parser.parse_args()

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
    global_briefing = None

    if global_data:
        print("\nLoading latest global briefing...")
        global_briefing = load_latest_global_briefing(script_dir)
        if global_briefing:
            print(f"  Week ending {global_briefing.get('week_ending', '?')}")
        else:
            print("  No global briefing found")

        print("Loading global coactivation state...")
        global_coactivation = load_coactivation_state(script_dir, "global")
        if global_coactivation:
            print(f"  State: week {global_coactivation.get('week_count', 0)}, "
                  f"last updated {global_coactivation.get('last_updated', '?')}")
        else:
            print("  No global coactivation state — edges default to PS co-membership")

        print("Computing global neuron positions...")
        global_positions = compute_neuron_positions(global_data, global_coactivation)
        print("Building global network...")
        global_net, global_node_tooltips, global_edge_tooltips = build_network(
            global_data, coactivation_state=global_coactivation, positions=global_positions
        )
        global_legend = build_global_legend_html(global_briefing)

    # --- US network ---
    print("\nLoading latest US briefing...")
    briefing = load_latest_briefing(script_dir)
    if briefing:
        print(f"  Week ending {briefing.get('week_ending', '?')}")
    else:
        print("  No briefing found")

    print("Loading US coactivation state...")
    coactivation_state = load_coactivation_state(script_dir, "us")
    if coactivation_state:
        print(f"  State: week {coactivation_state.get('week_count', 0)}, "
              f"last updated {coactivation_state.get('last_updated', '?')}")
    else:
        print("  No state — run: python coactivation_updater.py --bootstrap --scope us")

    print("Computing US neuron positions...")
    us_positions = compute_neuron_positions(us_data, coactivation_state)
    print("Building US neuron network...")
    us_net, us_node_tooltips, us_edge_tooltips = build_us_network(
        us_data, briefing=briefing, coactivation_state=coactivation_state,
        positions=us_positions
    )
    us_legend     = build_us_legend_html(briefing)
    us_asm_legend = build_us_asm_legend_html(briefing)
    ps_panel_data = build_ps_panel_data(briefing)
    if global_briefing:
        ps_panel_data.update(build_ps_panel_data(global_briefing))

    # --- Assembly networks ---
    ps_membership_edges = {
        "global_neuron":   build_ps_membership_edges(global_data)    if global_data else {},
        "us_neuron":       build_ps_membership_edges(us_data),
        "global_assembly": build_ps_ca_membership_edges(global_data) if global_data else {},
        "us_assembly":     build_ps_ca_membership_edges(us_data),
    }
    print(f"  Built PS membership edges for "
          f"{len(ps_membership_edges['us_neuron'])} US neuron / "
          f"{len(ps_membership_edges['us_assembly'])} US assembly / "
          f"{len(ps_membership_edges['global_neuron'])} global neuron / "
          f"{len(ps_membership_edges['global_assembly'])} global assembly PSs")

    print("Computing US assembly positions...")
    us_asm_positions = compute_ca_positions(us_data, coactivation_state)
    print("Building US assembly network...")
    us_asm_net, us_asm_node_tooltips, us_asm_edge_tooltips = build_assembly_network(
        us_data, briefing=briefing, is_us=True, coactivation_state=coactivation_state,
        positions=us_asm_positions
    )
    n_us_asm = len(us_data.get("canonical_vocabulary", {}).get("cell_assemblies", []))

    global_asm_net = global_asm_node_tooltips = global_asm_edge_tooltips = None
    global_asm_legend = None
    n_global_asm   = None
    if global_data:
        print("Computing global assembly positions...")
        global_asm_positions = compute_ca_positions(global_data, global_coactivation)
        print("Building global assembly network...")
        global_asm_net, global_asm_node_tooltips, global_asm_edge_tooltips = build_assembly_network(
            global_data, briefing=global_briefing, is_us=False,
            positions=global_asm_positions
        )
        n_global_asm = len(global_data.get("canonical_vocabulary", {}).get("cell_assemblies", []))
        global_asm_legend = build_global_asm_legend_html(global_briefing)

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
            global_asm_legend_html=global_asm_legend,
            us_asm_legend_html=us_asm_legend,
            n_global=n_global,
            n_us=n_us,
            n_global_asm=n_global_asm,
            n_us_asm=n_us_asm,
            has_global=True,
            ps_panel_data=ps_panel_data,
            ps_membership_edges=ps_membership_edges,
        )
    else:
        # US-only: scope toggle hidden, view type toggle still shown
        us_legend_visible     = us_legend.replace('display:none;', '', 1)
        us_asm_legend_visible = us_asm_legend.replace('display:none;', '', 1)
        build_combined_html(
            global_net=us_net,
            global_node_tooltips=us_node_tooltips,
            global_edge_tooltips=us_edge_tooltips,
            output_path=output_path,
            us_asm_net=us_asm_net,
            us_asm_node_tooltips=us_asm_node_tooltips,
            us_asm_edge_tooltips=us_asm_edge_tooltips,
            global_legend_html=us_legend_visible,
            us_asm_legend_html=us_asm_legend_visible,
            n_global=n_us,
            n_us_asm=n_us_asm,
            has_global=False,
            ps_panel_data=ps_panel_data,
            ps_membership_edges=ps_membership_edges,
        )

    print(f"Written to {output_path}")
    if not args.no_serve:
        serve_and_open(output_path, port=8766)


if __name__ == "__main__":
    main()
