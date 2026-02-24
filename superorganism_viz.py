"""
Superorganism Visualizer
Generates an interactive pyvis network graph from superorganism_model.json.

Nodes  = 17 prime movers, sized by rank, colored by hemisphere
Edges  = shared phase sequence participation, colored by excitatory/inhibitory valence
Opens  in default browser via localhost

Tooltip approach:
  - vis.js tooltip div is hidden via CSS (it renders HTML as text)
  - Custom tooltip overlay is injected and driven by vis.js hoverNode/hoverEdge events
  - network polling ensures listeners attach after vis.js fully initializes

Edge valence approach:
  - East vs West actors on COMPETITIVE phase sequences = INHIBITORY (red)
  - Same-hemisphere actors on any shared PS = EXCITATORY (green)
  - Mixed valence (some PS excitatory, some inhibitory) = AMBER
  - Tooltip shows per-PS breakdown with ⊕ / ⊖ symbols
"""

import json
import os
import webbrowser
import http.server
import threading
from pyvis.network import Network

# ---------------------------------------------------------------------------
# VISUAL CONSTANTS
# ---------------------------------------------------------------------------

HEMISPHERE_COLORS = {
    "West":      "#4a90d9",
    "East":      "#e05252",
    "Bridge":    "#f5a623",
    "Ancestral": "#7ed321",
}
HEMISPHERE_BORDER = {
    "West":      "#2c6fad",
    "East":      "#a83232",
    "Bridge":    "#c47d0e",
    "Ancestral": "#5aab12",
}

EXCITATORY_COLOR  = "#4caf50"   # green   — mutually amplifying
INHIBITORY_COLOR  = "#cf6679"   # muted red — opposing
MIXED_COLOR       = "#f5a623"   # amber   — some of each

BG_COLOR   = "#0d1117"
FONT_COLOR = "#e6edf3"

# Phase sequences that are inherently competitive:
# opposing actors on these sequences work AGAINST each other
COMPETITIVE_PS = {"PS-01", "PS-02", "PS-04", "PS-05", "PS-07", "PS-09"}
# PS-03 (Multiplanetary), PS-06 (Middle East — complex but no clear E/W split),
# PS-08 (Energy Transition), PS-10 (Alliance Reformation) default to excitatory/structural


def node_size(rank: int) -> int:
    return int(55 - (rank - 1) * 2.2)


def hemispheres_opposed(hemi_a: str, hemi_b: str) -> bool:
    return (hemi_a == "West" and hemi_b == "East") or \
           (hemi_a == "East" and hemi_b == "West")


def compute_ps_valences(person_a: dict, person_b: dict, shared_ids: set) -> list:
    """
    Returns list of (ps_id, ps_name, valence) where valence is +1 or -1.
    Uses hemisphere opposition + competitive PS rules.
    """
    hemi_a = person_a["superorganism"]["hemisphere"]
    hemi_b = person_b["superorganism"]["hemisphere"]
    opposed = hemispheres_opposed(hemi_a, hemi_b)

    ps_map = {ps["id"]: ps["name"]
              for ps in person_a["superorganism"].get("phase_sequences", [])}

    result = []
    for ps_id in sorted(shared_ids):
        name = ps_map.get(ps_id, ps_id)
        if opposed and ps_id in COMPETITIVE_PS:
            result.append((ps_id, name, -1))   # inhibitory
        else:
            result.append((ps_id, name, +1))   # excitatory
    return result


def edge_color_from_valences(valences: list) -> str:
    net   = sum(v for _, _, v in valences)
    total = len(valences)
    if net ==  total: return EXCITATORY_COLOR
    if net == -total: return INHIBITORY_COLOR
    return MIXED_COLOR


# ---------------------------------------------------------------------------
# BUILD NETWORK
# ---------------------------------------------------------------------------

def build_network(data: dict):
    """
    Returns (net, node_tooltips, edge_tooltips).
    No title= is set on any node or edge — vis.js tooltip is suppressed via CSS.
    """
    net = Network(
        height="100vh",
        width="100%",
        bgcolor=BG_COLOR,
        font_color=FONT_COLOR,
        directed=False,
        notebook=False,
    )

    people = data["superorganism_list"]
    node_tooltips = {}
    edge_tooltips = {}

    # --- Nodes ---
    for person in people:
        so         = person["superorganism"]
        hemisphere = so.get("hemisphere", "West")
        color      = HEMISPHERE_COLORS.get(hemisphere, "#888888")
        border     = HEMISPHERE_BORDER.get(hemisphere, "#555555")
        size       = node_size(person["rank"])

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

        node_tooltips[person["name"]] = (
            f"<div style='max-width:360px;font-family:sans-serif;"
            f"font-size:13px;color:#e6edf3;line-height:1.5'>"
            f"<div style='font-size:16px;font-weight:700;margin-bottom:2px'>{person['name']}</div>"
            f"<div style='color:#8b949e;font-size:11px;margin-bottom:10px'>"
            f"{so.get('primary_region','')} &nbsp;&middot;&nbsp; "
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

        # No title= — vis.js tooltip suppressed via CSS
        net.add_node(
            person["name"],
            label=person["name"],
            color={"background": color, "border": border,
                   "highlight": {"background": color, "border": "#ffffff"}},
            size=size,
            font={"size": 13, "color": FONT_COLOR, "face": "sans-serif"},
            borderWidth=2,
        )

    # --- Edges ---
    for i, a in enumerate(people):
        for b in people[i + 1:]:
            ps_a = {ps["id"] for ps in a["superorganism"].get("phase_sequences", [])}
            ps_b = {ps["id"] for ps in b["superorganism"].get("phase_sequences", [])}
            shared_ids = ps_a & ps_b
            if not shared_ids:
                continue

            valences = compute_ps_valences(a, b, shared_ids)
            ecolor   = edge_color_from_valences(valences)
            count    = len(valences)

            excit_count = sum(1 for _, _, v in valences if v > 0)
            inhib_count = sum(1 for _, _, v in valences if v < 0)

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

            ps_lines = "".join(
                f"<div style='margin-top:5px'>"
                f"<span style='font-size:14px;color:{'#4caf50' if v > 0 else '#cf6679'}'>"
                f"{'⊕' if v > 0 else '⊖'}</span>"
                f"&nbsp;<span style='color:#8b949e;font-size:11px'>{pid}</span>"
                f"&nbsp;{name}</div>"
                for pid, name, v in valences
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

    # --- Physics ---
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
          "springLength": 220,
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
# STATIC OVERLAYS
# ---------------------------------------------------------------------------

LEGEND_HTML = """
<div id="legend" style="
    position:fixed; top:20px; right:20px; z-index:9999;
    background:rgba(13,17,23,0.92); border:1px solid #30363d;
    border-radius:8px; padding:14px 18px; font-family:sans-serif;
    font-size:13px; color:#e6edf3; min-width:200px; line-height:1.8">
  <div style="font-size:14px;font-weight:700;margin-bottom:8px">Hemispheres</div>
  <div><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#4a90d9;margin-right:8px;vertical-align:middle"></span>West (US, Europe)</div>
  <div><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#e05252;margin-right:8px;vertical-align:middle"></span>East (China, Russia)</div>
  <div><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#f5a623;margin-right:8px;vertical-align:middle"></span>Bridge (India, Middle East)</div>
  <div><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#7ed321;margin-right:8px;vertical-align:middle"></span>Ancestral (Africa)</div>
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

TITLE_HTML = """
<div style="position:fixed;top:20px;left:20px;z-index:9999;font-family:sans-serif;color:#e6edf3">
  <div style="font-size:20px;font-weight:bold;letter-spacing:0.5px">Human Superorganism</div>
  <div style="font-size:12px;color:#8b949e;margin-top:2px">17 Prime Movers &nbsp;&middot;&nbsp; Feb 2026</div>
</div>
"""

TOOLTIP_CSS = """
<style>
/* Kill vis.js's own tooltip — it renders HTML as plain text */
div.vis-tooltip { display: none !important; }

#custom-tooltip {
  display: none;
  position: fixed;
  z-index: 10000;
  background: rgba(13, 17, 23, 0.96);
  border: 1px solid #30363d;
  border-radius: 8px;
  padding: 14px 16px;
  max-width: 380px;
  pointer-events: none;
  box-shadow: 0 8px 28px rgba(0,0,0,0.7);
}
</style>
"""

TOOLTIP_DIV = '<div id="custom-tooltip"></div>'


def build_tooltip_js(node_tooltips: dict, edge_tooltips: dict) -> str:
    node_js = json.dumps(node_tooltips, ensure_ascii=False)
    edge_js = json.dumps(edge_tooltips, ensure_ascii=False)

    return f"""
<script type="text/javascript">
(function() {{
  var nodeTooltips = {node_js};
  var edgeTooltips = {edge_js};
  var mouseX = 0, mouseY = 0;

  document.addEventListener('mousemove', function(e) {{
    mouseX = e.clientX;
    mouseY = e.clientY;
    var tip = document.getElementById('custom-tooltip');
    if (tip && tip.style.display === 'block') positionTooltip(tip);
  }});

  function positionTooltip(tip) {{
    var pad = 18, tw = tip.offsetWidth, th = tip.offsetHeight;
    var vw = window.innerWidth, vh = window.innerHeight;
    var x = mouseX + pad, y = mouseY - pad;
    if (x + tw > vw - pad) x = mouseX - tw - pad;
    if (y + th > vh - pad) y = vh - th - pad;
    if (y < pad) y = pad;
    tip.style.left = x + 'px';
    tip.style.top  = y + 'px';
  }}

  function showTooltip(html) {{
    var tip = document.getElementById('custom-tooltip');
    tip.innerHTML = html;
    tip.style.display = 'block';
    positionTooltip(tip);
  }}

  function hideTooltip() {{
    var tip = document.getElementById('custom-tooltip');
    if (tip) tip.style.display = 'none';
  }}

  // Poll until vis.js network object is ready, then attach listeners.
  // (window.load is unreliable — network may not have its event system ready yet.)
  var pollCount = 0;
  var poll = setInterval(function() {{
    pollCount++;
    if (typeof network !== 'undefined' && network.body && network.on) {{
      clearInterval(poll);
      attachListeners();
    }}
    if (pollCount > 100) clearInterval(poll);  // give up after 5s
  }}, 50);

  function attachListeners() {{
    network.on('hoverNode', function(p) {{
      var html = nodeTooltips[p.node];
      if (html) showTooltip(html);
    }});
    network.on('blurNode',  hideTooltip);

    network.on('hoverEdge', function(p) {{
      var edge = network.body.data.edges.get(p.edge);
      if (!edge) return;
      var html = edgeTooltips[edge.from + '|||' + edge.to]
              || edgeTooltips[edge.to   + '|||' + edge.from];
      if (html) showTooltip(html);
    }});
    network.on('blurEdge',  hideTooltip);
    network.on('dragStart', hideTooltip);
  }}
}})();
</script>
"""


# ---------------------------------------------------------------------------
# INJECT OVERLAYS
# ---------------------------------------------------------------------------

def inject_overlays(html_path: str, node_tooltips: dict, edge_tooltips: dict):
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    tooltip_js = build_tooltip_js(node_tooltips, edge_tooltips)

    html = html.replace("</head>", TOOLTIP_CSS + "</head>")
    html = html.replace(
        "</body>",
        TOOLTIP_DIV + LEGEND_HTML + TITLE_HTML + tooltip_js + "</body>"
    )

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# SERVE & OPEN
# ---------------------------------------------------------------------------

def serve_and_open(html_path: str, port: int = 8765):
    serve_dir = os.path.dirname(html_path)
    filename  = os.path.basename(html_path)
    os.chdir(serve_dir)

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args): pass

    server = http.server.HTTPServer(("localhost", port), QuietHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    url = f"http://localhost:{port}/{filename}"
    print(f"Serving at {url}")
    webbrowser.open(url)
    print("Press Ctrl+C to stop.\n")

    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.shutdown()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    input_path  = os.path.join(script_dir, "superorganism_model.json")
    output_path = os.path.join(script_dir, "superorganism_viz.html")

    print("=" * 60)
    print("SUPERORGANISM VISUALIZER")
    print("=" * 60)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    people = data["superorganism_list"]
    print(f"Building network: {len(people)} nodes...")
    net, node_tooltips, edge_tooltips = build_network(data)

    net.write_html(output_path)
    inject_overlays(output_path, node_tooltips, edge_tooltips)
    print(f"Graph written to {output_path}")

    serve_and_open(output_path)


if __name__ == "__main__":
    main()
