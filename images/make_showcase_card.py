"""Generate the PWLNet showcase card (LinkedIn / GitHub social preview / web).

Writes images/pwlnet_showcase.svg and images/pwlnet_showcase.png (2x, 2400x1254).
1200x627 is LinkedIn's 1.91:1 link-share size; it also works as a GitHub
social preview.

Run:  python images/make_showcase_card.py
"""

import math
from pathlib import Path

W, H = 1200, 627
FONT = "Inter, Segoe UI, Arial, sans-serif"
MONO = "JetBrains Mono, Consolas, Menlo, monospace"

BLUE, PURPLE, GREEN, AMBER = "#38BDF8", "#A78BFA", "#34D399", "#FBBF24"
TEXT, MUTED, DIM = "#F8FAFC", "#CBD5E1", "#94A3B8"


def text(x, y, s, size, fill=TEXT, weight=400, anchor="start", family=FONT, extra=""):
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    return (
        f'<text x="{x}" y="{y}"{a} font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}"{extra}>{s}</text>'
    )


def chip(x, y, label, color, size=13.5, pad=14, h=30, char_w=7.6, dashed=False):
    w = round(len(label) * char_w + 2 * pad)
    dash = ' stroke-dasharray="4 3"' if dashed else ""
    out = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h/2}" fill="{color}" fill-opacity="0.13" stroke="{color}" stroke-opacity="0.7"{dash}/>',
        text(x + w / 2, y + h / 2 + size * 0.36, label, size, TEXT, 650, "middle"),
    ]
    return out, w


def build():
    s = [
        f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" fill="none" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="PWLNet: optimization-aware neural surrogates for process systems engineering">',
        "<defs>",
        '<linearGradient id="bg" x1="0" y1="0" x2="1200" y2="627" gradientUnits="userSpaceOnUse">'
        '<stop stop-color="#020617"/><stop offset="0.5" stop-color="#0F172A"/><stop offset="1" stop-color="#111827"/></linearGradient>',
        f'<linearGradient id="rainbow" x1="64" y1="0" x2="420" y2="0" gradientUnits="userSpaceOnUse">'
        f'<stop stop-color="{BLUE}"/><stop offset="0.4" stop-color="{PURPLE}"/><stop offset="0.75" stop-color="{GREEN}"/><stop offset="1" stop-color="{AMBER}"/></linearGradient>',
        f'<linearGradient id="bar" x1="0" y1="0" x2="1200" y2="0" gradientUnits="userSpaceOnUse">'
        f'<stop stop-color="{BLUE}"/><stop offset="0.4" stop-color="{PURPLE}"/><stop offset="0.75" stop-color="{GREEN}"/><stop offset="1" stop-color="{AMBER}"/></linearGradient>',
        '<linearGradient id="panel" x1="0" y1="0" x2="0" y2="1">'
        '<stop stop-color="#1E293B" stop-opacity="0.95"/><stop offset="1" stop-color="#0F172A" stop-opacity="0.92"/></linearGradient>',
        '<radialGradient id="glow" cx="920" cy="300" r="420" gradientUnits="userSpaceOnUse">'
        f'<stop stop-color="{PURPLE}" stop-opacity="0.18"/><stop offset="1" stop-color="{PURPLE}" stop-opacity="0"/></radialGradient>',
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">'
        '<feDropShadow dx="0" dy="14" stdDeviation="16" flood-color="#000000" flood-opacity="0.45"/></filter>',
        "</defs>",
        f'<rect width="{W}" height="{H}" fill="url(#bg)"/>',
        f'<rect width="{W}" height="{H}" fill="url(#glow)"/>',
        '<g opacity="0.06" stroke="#CBD5E1">'
        + "<path d=\"" + " ".join(f"M0 {y}H{W}" for y in range(60, H, 60)) + "\"/>"
        + "<path d=\"" + " ".join(f"M{x} 0V{H}" for x in range(60, W, 60)) + "\"/></g>",
        f'<rect x="0" y="0" width="{W}" height="5" fill="url(#bar)"/>',
    ]

    # ---- Left column: identity ----
    L = 64
    c, w = chip(L, 52, "Open-source Python library", BLUE, size=12.5, h=28, char_w=7.0)
    s += c
    s.append(text(L - 3, 158, "PWLNet", 86, "url(#rainbow)", 850, extra=' letter-spacing="-1.5"'))
    s.append(text(L, 206, "Optimization-aware neural surrogates", 27, TEXT, 700))
    s.append(text(L, 240, "for process systems engineering", 27, TEXT, 700))
    s.append(text(L, 278, "Train compact networks, embed them exactly in Pyomo,", 16, MUTED))
    s.append(text(L, 301, "and solve for certified process decisions.", 16, MUTED))

    # Pipeline
    y = 334
    x = L
    steps = [("Train", BLUE), ("Prune", PURPLE), ("Reformulate", GREEN), ("Solve", AMBER)]
    for k, (lab, col) in enumerate(steps):
        c, w = chip(x, y, lab, col, size=14, h=32, char_w=8.2)
        s += c
        x += w
        if k < len(steps) - 1:
            s.append(f'<path d="M{x+8} {y+16}H{x+24}" stroke="#64748B" stroke-width="2" stroke-linecap="round"/>')
            s.append(f'<path d="M{x+19} {y+11}L{x+25} {y+16}L{x+19} {y+21}" stroke="#94A3B8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>')
            x += 34

    # Workflows
    s.append(text(L, 410, "WORKFLOWS", 11.5, DIM, 750, extra=' letter-spacing="1.6"'))
    x = L
    for lab, col, dashed in [("ReLU-MIP", BLUE, False), ("tanh-PWL", PURPLE, False), ("SOSX branch + LP", GREEN, False), ("physics-informed", AMBER, True)]:
        c, w = chip(x, 422, lab, col, size=13, h=28, char_w=7.4, dashed=dashed)
        s += c
        x += w + 10

    # Stack + benchmarks
    s.append(text(L, 486, "STACK", 11.5, DIM, 750, extra=' letter-spacing="1.6"'))
    s.append(text(L, 510, "TensorFlow · Keras · Pyomo · OMLT · Gurobi · HiGHS · Ipopt · MOSEK", 15, MUTED, 500))
    s.append(text(L, 546, "BENCHMARKS", 11.5, DIM, 750, extra=' letter-spacing="1.6"'))
    s.append(text(L, 570, "Williams-Otto · Hydroformylation · Van de Vusse", 15, MUTED, 500))

    # ---- Right column: visual ----
    px, py, pw, ph = 726, 52, 410, 523
    s.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="24" fill="url(#panel)" stroke="#334155" filter="url(#shadow)"/>')
    s.append(text(px + 24, py + 36, "tanh  →  piecewise-linear", 15, TEXT, 750))
    s.append(text(px + 24, py + 56, "exact MIP encoding of the network", 12.5, DIM))

    # Plot area
    ax0, ax1, ay0, ay1 = px + 34, px + pw - 26, py + 82, py + 300  # x range, y top/bottom
    xr, yr = 4.0, 1.15

    def X(v):
        return ax0 + (v + xr) / (2 * xr) * (ax1 - ax0)

    def Y(v):
        return ay1 - (v + yr) / (2 * yr) * (ay1 - ay0)

    for gv in (-1, 0, 1):
        s.append(f'<path d="M{ax0} {Y(gv):.1f}H{ax1}" stroke="#334155" stroke-width="1" {"" if gv == 0 else "stroke-dasharray=\"3 4\""}/>')
    s.append(f'<path d="M{X(0):.1f} {ay0}V{ay1}" stroke="#334155" stroke-width="1"/>')

    smooth = " ".join(
        f"{'M' if i == 0 else 'L'}{X(v):.1f} {Y(math.tanh(v)):.1f}"
        for i, v in enumerate([-xr + 2 * xr * i / 160 for i in range(161)])
    )
    s.append(f'<path d="{smooth}" stroke="{MUTED}" stroke-width="2" stroke-dasharray="5 5" stroke-opacity="0.75"/>')

    bps = [-4, -2, -1.1, -0.45, 0, 0.45, 1.1, 2, 4]
    pwl = " ".join(f"{'M' if i == 0 else 'L'}{X(b):.1f} {Y(math.tanh(b)):.1f}" for i, b in enumerate(bps))
    s.append(f'<path d="{pwl}" stroke="{PURPLE}" stroke-width="3.2" stroke-linejoin="round" stroke-linecap="round"/>')
    for b in bps:
        s.append(f'<circle cx="{X(b):.1f}" cy="{Y(math.tanh(b)):.1f}" r="4.6" fill="#0F172A" stroke="{GREEN}" stroke-width="2.2"/>')

    # Legend
    ly = ay1 + 30
    s.append(f'<path d="M{px+34} {ly}H{px+60}" stroke="{MUTED}" stroke-width="2" stroke-dasharray="5 5"/>')
    s.append(text(px + 68, ly + 4.5, "tanh", 12.5, DIM))
    s.append(f'<path d="M{px+124} {ly}H{px+150}" stroke="{PURPLE}" stroke-width="3.2" stroke-linecap="round"/>')
    s.append(text(px + 158, ly + 4.5, "PWL surrogate", 12.5, DIM))
    s.append(f'<circle cx="{px+270}" cy="{ly}" r="4.6" fill="#0F172A" stroke="{GREEN}" stroke-width="2.2"/>')
    s.append(text(px + 282, ly + 4.5, "breakpoints", 12.5, DIM))

    # Formulation box
    fx, fy, fw, fh = px + 20, ly + 26, pw - 40, 140
    s.append(f'<rect x="{fx}" y="{fy}" width="{fw}" height="{fh}" rx="16" fill="#020617" fill-opacity="0.75" stroke="#334155"/>')
    rows = [
        ("max", f'<tspan fill="{TEXT}">profit(x, u)</tspan>'),
        ("s.t.", f'<tspan fill="{PURPLE}">y = NN(x, u)</tspan>'),
        ("", f'<tspan fill="{DIM}">bounds on x, u, y</tspan>'),
    ]
    for k, (kw, body) in enumerate(rows):
        yy = fy + 34 + k * 26
        if kw:
            s.append(text(fx + 20, yy, kw, 15, AMBER, 700, family=MONO))
        s.append(text(fx + 76, yy, body, 15, TEXT, 500, family=MONO))
    s.append(f'<path d="M{fx+20} {fy+104}H{fx+fw-20}" stroke="#334155"/>')
    s.append(text(fx + fw / 2, fy + 125, "→  MILP · MIQP · MISOCP · NLP", 13.5, GREEN, 700, "middle"))

    # Footer link
    s.append(text(W - 64, H - 18, "github.com/SunoySanyal/PWLNet", 13, DIM, 600, "end"))
    s.append("</svg>")
    return "\n".join(s) + "\n"


if __name__ == "__main__":
    here = Path(__file__).parent
    svg_path = here / "pwlnet_showcase.svg"
    svg_path.write_text(build(), encoding="utf-8")
    import cairosvg

    cairosvg.svg2png(url=str(svg_path), write_to=str(here / "pwlnet_showcase.png"), output_width=2 * W, output_height=2 * H)
    print(f"wrote {svg_path} and pwlnet_showcase.png")
