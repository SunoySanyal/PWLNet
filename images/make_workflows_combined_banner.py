"""Generate images/workflows_combined_banner.svg.

Layout: shared pipeline (data -> train -> network object) across the top,
fanning out into three vertical workflow columns that merge into one
solver-ready output bar at the bottom.

Run:  python images/make_workflows_combined_banner.py
"""

from pathlib import Path

W, H = 1280, 720
FONT = "Inter, Segoe UI, Arial, sans-serif"

# Column geometry
COL_X0, COL_W, COL_GAP = 56, 372, 26
COL_TOP, COL_BOT = 222, 608
STEP_Y0, STEP_H, STEP_GAP = 298, 48, 14
BUS_IN_Y, BUS_OUT_Y = 204, 624

AMBER, AMBER_LIGHT = "#FBBF24", "#FEF3C7"

# Steps are (title, subtitle) or (title, subtitle, "optional"[, "amber"]).
# Optional steps get a dashed border + pill; physics-informed training is
# shared by every path, so it is step 1 everywhere and drawn in amber.
PHYSICS = ("Physics-informed training", "ODE residual + bound penalties", "optional", "amber")
MIP_CLASSES = "MILP / MIQP / MISOCP"

WORKFLOWS = [
    dict(
        name="ReLU-MIP", tag="exact mixed-integer encoding",
        c="#38BDF8", c2="#0EA5E9", dark="#082F49", light="#E0F2FE",
        glyph=['<path d="M0 22 L16 22 L34 2"/>'],  # ReLU
        steps=[
            PHYSICS,
            ("Prune", "sparse ReLU layers", "optional"),
            ("Tighten bounds", "interval / LP bounds"),
            ("OMLT block", "big-M Pyomo encoding"),
            ("Solve MIP", MIP_CLASSES),
        ],
    ),
    dict(
        name="tanh-PWL", tag="piecewise-linear smooth nets",
        c="#A78BFA", c2="#7C3AED", dark="#2E1065", light="#EDE9FE",
        glyph=['<path d="M0 22 L10 20 L14 14 L20 8 L24 3 L34 1"/>'],  # PWL tanh
        steps=[
            PHYSICS,
            ("Prune", "compact tanh network", "optional"),
            ("PWL tanh", "breakpoint approximation"),
            ("PWL-aware training", "train through PWL: no embedding shift", "optional"),
            ("big-M / SOS2 solve", MIP_CLASSES),
        ],
    ),
    dict(
        name="SOSX", tag="custom branch + LP engine",
        c="#34D399", c2="#059669", dark="#064E3B", light="#D1FAE5",
        # small branch-and-bound tree
        glyph=[
            '<path d="M17 4 L6 20 M17 4 L28 20"/>',
            '<circle cx="17" cy="4" r="3.2" fill="currentColor"/>',
            '<circle cx="6" cy="20" r="3.2" fill="currentColor"/>',
            '<circle cx="28" cy="20" r="3.2" fill="currentColor"/>',
        ],
        steps=[
            PHYSICS,
            ("SOS2 blocks", "λ-weights per neuron"),
            ("Relax", "drop SOS2, solve LP"),
            ("Branch", "split the λ window"),
            ("Close the gap", "certified bound"),
        ],
    ),
]

TOP_NODES = [
    ("Process Data", "clean, scale, split", "data"),
    ("Train Surrogate", "ReLU or tanh network", "nn"),
    ("Network Object", "weights, bounds, scalers", "obj"),
]


def col_x(i):
    return COL_X0 + i * (COL_W + COL_GAP)


def icon(kind, cx, cy, color, light):
    out = [f'<circle cx="{cx}" cy="{cy}" r="19" fill="{color}" fill-opacity="0.18" stroke="{color}"/>']
    if kind == "data":
        out += [
            f'<ellipse cx="{cx}" cy="{cy-7}" rx="10" ry="3.6" stroke="{light}" stroke-width="1.7"/>',
            f'<path d="M{cx-10} {cy-7}V{cy+7}C{cx-10} {cy+10} {cx-5} {cy+12} {cx} {cy+12}C{cx+5} {cy+12} {cx+10} {cy+10} {cx+10} {cy+7}V{cy-7}" stroke="{light}" stroke-width="1.7"/>',
            f'<path d="M{cx-10} {cy}C{cx-10} {cy+3} {cx-5} {cy+5} {cx} {cy+5}C{cx+5} {cy+5} {cx+10} {cy+3} {cx+10} {cy}" stroke="{light}" stroke-width="1.4"/>',
        ]
    elif kind == "nn":
        pts = [(-9, -7), (-9, 7), (0, 0), (9, -8), (9, 8)]
        out.append(
            f'<path d="M{cx-9} {cy-7}L{cx} {cy}L{cx-9} {cy+7}M{cx} {cy}L{cx+9} {cy-8}M{cx} {cy}L{cx+9} {cy+8}" stroke="{light}" stroke-width="1.3"/>'
        )
        out += [f'<circle cx="{cx+dx}" cy="{cy+dy}" r="3.2" fill="{light}"/>' for dx, dy in pts]
    else:  # obj: PWL curve on an axis
        out += [
            f'<path d="M{cx-10} {cy+8}L{cx-3} {cy-8}L{cx+3} {cy+3}L{cx+10} {cy-5}" stroke="{light}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>',
            f'<path d="M{cx-12} {cy+11}H{cx+12}" stroke="{light}" stroke-width="1.6" stroke-linecap="round"/>',
        ]
    return out


def arrow_down(x, y0, y1, color, width=2):
    return [
        f'<path d="M{x} {y0}V{y1-1}" stroke="{color}" stroke-width="{width}" stroke-linecap="round"/>',
        f'<path d="M{x-5} {y1-7}L{x} {y1}L{x+5} {y1-7}" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"/>',
    ]


def build():
    s = []
    s.append(
        f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" fill="none" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc">'
    )
    s.append('  <title id="title">PWLNet combined workflow map</title>')
    s.append(
        '  <desc id="desc">Process data is trained into a network object, which fans out into three workflows '
        '(ReLU-MIP, tanh-PWL, SOSX), each with optional physics-informed training and pruning, that all produce a solver-ready decision model.</desc>'
    )
    s.append("  <defs>")
    s.append(
        '    <linearGradient id="bg" x1="0" y1="0" x2="1280" y2="720" gradientUnits="userSpaceOnUse">'
        '<stop stop-color="#020617"/><stop offset="0.45" stop-color="#0F172A"/><stop offset="1" stop-color="#111827"/></linearGradient>'
    )
    s.append(
        '    <linearGradient id="panel" x1="0" y1="0" x2="0" y2="1">'
        '<stop stop-color="#1E293B" stop-opacity="0.96"/><stop offset="1" stop-color="#0F172A" stop-opacity="0.92"/></linearGradient>'
    )
    s.append(
        # userSpaceOnUse: objectBoundingBox gradients don't paint zero-height lines
        f'    <linearGradient id="rainbow" x1="{COL_X0 + COL_W // 2}" y1="0" x2="{W - COL_X0 - COL_W // 2}" y2="0" gradientUnits="userSpaceOnUse">'
        '<stop stop-color="#38BDF8"/><stop offset="0.35" stop-color="#A78BFA"/><stop offset="0.68" stop-color="#34D399"/><stop offset="1" stop-color="#FBBF24"/></linearGradient>'
    )
    for i, wf in enumerate(WORKFLOWS):
        s.append(
            f'    <linearGradient id="g{i}" x1="0" y1="0" x2="1" y2="0"><stop stop-color="{wf["c"]}"/><stop offset="1" stop-color="{wf["c2"]}"/></linearGradient>'
        )
    s.append(
        '    <filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">'
        '<feDropShadow dx="0" dy="10" stdDeviation="12" flood-color="#000000" flood-opacity="0.38"/></filter>'
    )
    s.append("  </defs>")

    # Background + faint grid
    s.append(f'  <rect width="{W}" height="{H}" rx="34" fill="url(#bg)"/>')
    s.append('  <g opacity="0.07" stroke="#CBD5E1">')
    s.append("    <path d=\"" + " ".join(f"M0 {y}H{W}" for y in range(90, H, 90)) + "\"/>")
    s.append("    <path d=\"" + " ".join(f"M{x} 0V{H}" for x in range(100, W, 120)) + "\"/>")
    s.append("  </g>")

    # Title
    s.append(
        f'  <text x="640" y="54" text-anchor="middle" font-family="{FONT}" font-size="36" font-weight="800" fill="#F8FAFC">PWLNet Workflow Map</text>'
    )
    s.append(
        f'  <text x="640" y="82" text-anchor="middle" font-family="{FONT}" font-size="14" fill="#94A3B8">One trained surrogate, three optimization-aware paths, one solver-ready decision model</text>'
    )

    # Top pipeline
    node_w, node_h, node_gap, node_y = 270, 64, 70, 106
    total = 3 * node_w + 2 * node_gap
    nx0 = (W - total) // 2
    top_colors = [("#38BDF8", "#E0F2FE"), ("#A78BFA", "#EDE9FE"), ("#34D399", "#D1FAE5")]
    for k, (title, sub, kind) in enumerate(TOP_NODES):
        x = nx0 + k * (node_w + node_gap)
        c, light = top_colors[k]
        s.append(
            f'  <rect x="{x}" y="{node_y}" width="{node_w}" height="{node_h}" rx="18" fill="url(#panel)" stroke="{c}" stroke-opacity="0.55" filter="url(#shadow)"/>'
        )
        s += ["  " + e for e in icon(kind, x + 38, node_y + node_h // 2, c, light)]
        s.append(
            f'  <text x="{x+70}" y="{node_y+29}" font-family="{FONT}" font-size="15" font-weight="750" fill="#F8FAFC">{title}</text>'
        )
        s.append(f'  <text x="{x+70}" y="{node_y+48}" font-family="{FONT}" font-size="12" fill="#94A3B8">{sub}</text>')
        if k < 2:
            ax0, ax1, ay = x + node_w + 12, x + node_w + node_gap - 12, node_y + node_h // 2
            s.append(f'  <path d="M{ax0} {ay}H{ax1-2}" stroke="#64748B" stroke-width="2.4" stroke-linecap="round"/>')
            s.append(
                f'  <path d="M{ax1-9} {ay-7}L{ax1} {ay}L{ax1-9} {ay+7}" stroke="#94A3B8" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>'
            )

    # Fan-out bus from the network object to the columns
    obj_cx = nx0 + 2 * (node_w + node_gap) + node_w // 2
    centers = [col_x(i) + COL_W // 2 for i in range(len(WORKFLOWS))]
    s.append(f'  <path d="M{obj_cx} {node_y+node_h}V{BUS_IN_Y}" stroke="#64748B" stroke-width="2.2"/>')
    s.append(f'  <path d="M{centers[0]} {BUS_IN_Y}H{centers[-1]}" stroke="url(#rainbow)" stroke-width="2.4" stroke-opacity="0.85"/>')
    s.append(f'  <circle cx="{obj_cx}" cy="{BUS_IN_Y}" r="4" fill="#CBD5E1"/>')
    for i, cx in enumerate(centers):
        s += ["  " + e for e in arrow_down(cx, BUS_IN_Y, COL_TOP, WORKFLOWS[i]["c"], 2.2)]

    # Columns
    for i, wf in enumerate(WORKFLOWS):
        x, c = col_x(i), wf["c"]
        s.append(f"  <!-- {wf['name']} -->")
        s.append(
            f'  <rect x="{x}" y="{COL_TOP}" width="{COL_W}" height="{COL_BOT-COL_TOP}" rx="22" fill="url(#panel)" stroke="{c}" stroke-opacity="0.5" filter="url(#shadow)"/>'
        )
        s.append(f'  <rect x="{x+22}" y="{COL_TOP}" width="{COL_W-44}" height="3" rx="1.5" fill="url(#g{i})"/>')
        # Header
        hx, hy, hw, hh = x + 14, COL_TOP + 16, COL_W - 28, 52
        s.append(
            f'  <rect x="{hx}" y="{hy}" width="{hw}" height="{hh}" rx="14" fill="{wf["dark"]}" fill-opacity="0.85" stroke="{c}" stroke-opacity="0.6"/>'
        )
        s.append(
            f'  <text x="{hx+16}" y="{hy+23}" font-family="{FONT}" font-size="17" font-weight="800" fill="#F8FAFC">{wf["name"]}</text>'
        )
        s.append(f'  <text x="{hx+16}" y="{hy+41}" font-family="{FONT}" font-size="11.5" fill="{wf["light"]}" fill-opacity="0.8">{wf["tag"]}</text>')
        gx, gy = hx + hw - 50, hy + 14
        s.append(
            f'  <g transform="translate({gx} {gy})" color="{c}" stroke="{c}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">'
            + "".join(wf["glyph"]) + "</g>"
        )
        # Steps
        sx, sw = x + 14, COL_W - 28
        for j, step in enumerate(wf["steps"]):
            title, sub = step[0], step[1]
            optional = len(step) > 2
            sc, slight = (AMBER, AMBER_LIGHT) if len(step) > 3 else (c, wf["light"])
            sy = STEP_Y0 + j * (STEP_H + STEP_GAP)
            dash = ' stroke-dasharray="5 4"' if optional else ""
            s.append(
                f'  <rect x="{sx}" y="{sy}" width="{sw}" height="{STEP_H}" rx="12" fill="#0B1222" stroke="{sc}" stroke-opacity="{0.7 if optional else 0.35}"{dash}/>'
            )
            bx, by = sx + 24, sy + STEP_H // 2
            s.append(f'  <circle cx="{bx}" cy="{by}" r="12" fill="{sc}" fill-opacity="0.16" stroke="{sc}" stroke-opacity="0.8"/>')
            s.append(
                f'  <text x="{bx}" y="{by+4.5}" text-anchor="middle" font-family="{FONT}" font-size="12" font-weight="800" fill="{slight}">{j+1}</text>'
            )
            s.append(
                f'  <text x="{sx+48}" y="{sy+21}" font-family="{FONT}" font-size="14" font-weight="750" fill="#F8FAFC">{title}</text>'
            )
            s.append(f'  <text x="{sx+48}" y="{sy+37}" font-family="{FONT}" font-size="11.5" fill="#94A3B8">{sub}</text>')
            if optional:
                pw, ph = 62, 20
                px, py = sx + sw - pw - 12, sy + (STEP_H - ph) // 2
                s.append(f'  <rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="10" fill="{sc}" fill-opacity="0.14" stroke="{sc}" stroke-opacity="0.6"/>')
                s.append(
                    f'  <text x="{px + pw / 2}" y="{py + 14}" text-anchor="middle" font-family="{FONT}" font-size="10.5" font-weight="700" fill="{slight}">{step[2]}</text>'
                )
            if j < len(wf["steps"]) - 1:
                s += ["  " + e for e in arrow_down(bx, sy + STEP_H + 2, sy + STEP_H + STEP_GAP - 2, c, 1.8)]

    # Merge bus into shared output
    out_w, out_h, out_y = 640, 44, 642
    out_x = (W - out_w) // 2
    for i, cx in enumerate(centers):
        s.append(f'  <path d="M{cx} {COL_BOT}V{BUS_OUT_Y}" stroke="{WORKFLOWS[i]["c"]}" stroke-width="2.2" stroke-opacity="0.85"/>')
    s.append(f'  <path d="M{centers[0]} {BUS_OUT_Y}H{centers[-1]}" stroke="url(#rainbow)" stroke-width="2.4" stroke-opacity="0.85"/>')
    s += ["  " + e for e in arrow_down(W // 2, BUS_OUT_Y, out_y, "#CBD5E1", 2.2)]
    s.append(
        f'  <rect x="{out_x}" y="{out_y}" width="{out_w}" height="{out_h}" rx="22" fill="#020617" fill-opacity="0.85" stroke="url(#rainbow)" stroke-width="1.6"/>'
    )
    s.append(
        f'  <text x="{W//2}" y="{out_y+28}" text-anchor="middle" font-family="{FONT}" font-size="15" font-weight="750" fill="#F8FAFC">Solver-ready decision model  ·  MILP / MIQP / MISOCP / NLP in Pyomo</text>'
    )

    s.append("</svg>")
    return "\n".join(s) + "\n"


if __name__ == "__main__":
    out = Path(__file__).with_name("workflows_combined_banner.svg")
    out.write_text(build(), encoding="utf-8")
    print(f"wrote {out}")
