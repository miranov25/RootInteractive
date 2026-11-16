"""A RootInteractive dashboard of the ToyMC tables (collisions, tracks, V0s, cascades) joined by row numbers.

In plain words. The page shows figures in tabs by category (Collisions, Tracks, V0s, Cascades, Consistency). Every
figure of a track, a V0 or a cascade reads its parents through joins: a track its collision; a V0 its positive and
negative track and their collisions; a cascade its V0, that V0's tracks and collisions, and its bachelor track.
The "Consistency" tab compares the stored mass of a V0 and of a cascade with the mass computed in the browser from
the joined tracks; the two must agree. RootInteractive's code is not changed: the page is built by bokehDrawSA from an
ordinary page description.

Stage A of the plan of 2026-11-16 (display only): the page has no selection sliders yet.

Words used.
  role      The same table seen through one pointer. The tracks appear as Tracks (all tracks), Pos (the positive
            track of a V0), Neg (its negative track) and Bachelor (the third track of a cascade), as in the
            AliasDataFrame views of toy_mc_events.make_alias_dataframes.
  prefix    Every column on the page carries the prefix of its role (pos_pt, neg_pt, ...). The page needs it: a join
            reads a column by its plain name, and the ToyMC tables share names (pt, mass, collisionIdx).
  join      The browser model CDSJoin. A join by row number: the child row holds the row number of its parent.

Known limit of this stage: a table that is used in several roles is sent once per role (the tracks four times, the
collisions three times), each with only the columns its role needs.

Use:
    python toy_mc_joins_dashboard.py --n-collisions 2000 --output /tmp/toy_mc_joins_dashboard.html
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from RootInteractive.Tools.generators.toy_mc_events import (
        CASCADE_DAUGHTERS, MASS, V0_DAUGHTERS, generate_toy_mc_events)
except ImportError:                                    # started inside this directory
    from toy_mc_events import CASCADE_DAUGHTERS, MASS, V0_DAUGHTERS, generate_toy_mc_events

# role -> (ToyMC table, prefix, the columns sent: page name suffix = ToyMC column name)
ROLES = {
    "Collisions": ("collisions", "coll_", ("vz", "nTracks")),
    "PosCollisions": ("collisions", "posColl_", ("vz",)),
    "NegCollisions": ("collisions", "negColl_", ("vz",)),
    "Tracks": ("tracks", "trk_", ("pt", "eta", "phi", "collisionIdx")),
    "Pos": ("tracks", "pos_", ("pt", "eta", "phi", "collisionIdx")),
    "Neg": ("tracks", "neg_", ("pt", "eta", "phi", "collisionIdx")),
    "Bachelor": ("tracks", "bach_", ("pt", "eta", "phi")),
    "V0s": ("v0s", "v0_", ("mass", "pdgMass", "pdg", "pt", "px", "py", "pz", "posTrackIdx", "negTrackIdx")),
    "Cascades": ("cascades", "casc_", ("mass", "pdgMass", "pdg", "pt", "v0Idx", "bachelorTrackIdx")),
}
MAIN = "Cascades"                                      # the main table of the page; every other role is a named table

# name -> (child, parent, pointer column of the child). A child or parent may itself be a join.
JOINS = {
    "TrkJ": ("Tracks", "Collisions", "trk_collisionIdx"),          # a track and its collision
    "PosJ": ("Pos", "PosCollisions", "pos_collisionIdx"),          # a positive track and its collision
    "NegJ": ("Neg", "NegCollisions", "neg_collisionIdx"),          # a negative track and its collision
    "V0P": ("V0s", "PosJ", "v0_posTrackIdx"),                      # a V0, its positive track, that track's collision
    "V0J": ("V0P", "NegJ", "v0_negTrackIdx"),                      # ... and its negative track with its collision
    "CascV": (MAIN, "V0J", "casc_v0Idx"),                          # a cascade and everything of its V0
    "CascJ": ("CascV", "Bachelor", "casc_bachelorTrackIdx"),       # ... and its bachelor track
}

_P4 = lambda p, m: {                                   # the four-momentum of a track from pt, eta, phi and a mass
    "E": f"sqrt({m}**2 + ({p}pt*cosh({p}eta))**2)", "px": f"{p}pt*cos({p}phi)", "py": f"{p}pt*sin({p}phi)",
    "pz": f"{p}pt*sinh({p}eta)"}
_POS, _NEG, _BACH = _P4("pos_", "v0_massPos"), _P4("neg_", "v0_massNeg"), _P4("bach_", "casc_massBachelor")
# The derived columns, computed in the browser on the join named as context. The mass formulas are those of the
# generator (toy_mc_events: p4_from_pt_eta_phi, inv_mass, add_v0, add_cascade).
ALIASES = [
    {"name": "trk_tgl", "expr": "sinh(trk_eta)", "context": "TrkJ"},
    {"name": "v0_deltaMass", "expr": "v0_mass - v0_pdgMass", "context": "V0J"},
    {"name": "v0_tgl", "expr": "v0_pz / v0_pt", "context": "V0J"},
    {"name": "v0_invPt", "expr": "1 / v0_pt", "context": "V0J"},
    {"name": "v0_posPtOverPt", "expr": "pos_pt / v0_pt", "context": "V0J"},
    {"name": "v0_collisionVzDiff", "expr": "posColl_vz - negColl_vz", "context": "V0J"},
    {"name": "v0_massFromTracks", "context": "V0J",
     "expr": f"sqrt(({_POS['E']} + {_NEG['E']})**2 - ({_POS['px']} + {_NEG['px']})**2"
             f" - ({_POS['py']} + {_NEG['py']})**2 - ({_POS['pz']} + {_NEG['pz']})**2)"},
    {"name": "v0_massDiff", "expr": "v0_massFromTracks - v0_mass", "context": "V0J"},
    {"name": "casc_deltaMass", "expr": "casc_mass - casc_pdgMass", "context": "CascJ"},
    {"name": "casc_massFromV0AndBachelor", "context": "CascJ",
     "expr": f"sqrt((sqrt(v0_mass**2 + v0_px**2 + v0_py**2 + v0_pz**2) + {_BACH['E']})**2 - (v0_px + {_BACH['px']})**2"
             f" - (v0_py + {_BACH['py']})**2 - (v0_pz + {_BACH['pz']})**2)"},
    {"name": "casc_massDiff", "expr": "casc_massFromV0AndBachelor - casc_mass", "context": "CascJ"},
]

# name -> (column, source, number of bins, low, high, axis title, what is counted)
HISTOGRAMS = {
    "hCollVz": ("coll_vz", "Collisions", 60, -30.0, 30.0, "collision vz (cm)", "collisions"),
    "hCollNTracks": ("coll_nTracks", "Collisions", 40, 0.0, 40.0, "tracks per collision", "collisions"),
    "hTrkPt": ("trk_pt", "TrkJ", 50, 0.0, 10.0, "track pt (GeV/c)", "tracks"),
    "hTrkTgl": ("trk_tgl", "TrkJ", 50, -1.5, 1.5, "track pz/pt", "tracks"),
    "hTrkCollVz": ("coll_vz", "TrkJ", 60, -30.0, 30.0, "Collision.vz of the track (cm)", "tracks"),
    "hV0Mass": ("v0_mass", "V0J", 50, 0.0, 3.0, "V0 mass (GeV/c2)", "V0s"),
    "hV0DeltaMass": ("v0_deltaMass", "V0J", 50, -0.1, 0.1, "V0 mass minus nominal mass (GeV/c2)", "V0s"),
    "hV0PosPt": ("pos_pt", "V0J", 50, 0.0, 10.0, "Pos.pt (GeV/c)", "V0s"),
    "hV0NegPt": ("neg_pt", "V0J", 50, 0.0, 10.0, "Neg.pt (GeV/c)", "V0s"),
    "hV0PosCollVz": ("posColl_vz", "V0J", 60, -30.0, 30.0, "Pos.Collision.vz (cm)", "V0s"),
    "hV0Tgl": ("v0_tgl", "V0J", 50, -1.5, 1.5, "V0 pz/pt", "V0s"),
    "hCascMass": ("casc_mass", "CascJ", 50, 1.0, 4.0, "cascade mass (GeV/c2)", "cascades"),
    "hCascDeltaMass": ("casc_deltaMass", "CascJ", 50, -0.1, 0.1, "cascade mass minus nominal mass (GeV/c2)", "cascades"),
    "hCascV0Mass": ("v0_mass", "CascJ", 50, 0.0, 3.0, "V0.mass of the cascade (GeV/c2)", "cascades"),
    "hCascBachPt": ("bach_pt", "CascJ", 50, 0.0, 10.0, "Bachelor.pt (GeV/c)", "cascades"),
    "hCascPosCollVz": ("posColl_vz", "CascJ", 60, -30.0, 30.0, "V0.Pos.Collision.vz (cm)", "cascades"),
    "hV0MassDiff": ("v0_massDiff", "V0J", 50, -1e-9, 1e-9, "V0: mass from Pos and Neg minus stored mass (GeV/c2)", "V0s"),
    "hCascMassDiff": ("casc_massDiff", "CascJ", 50, -1e-9, 1e-9,
                      "cascade: mass from V0 and Bachelor minus stored mass (GeV/c2)", "cascades"),
    "hV0CollVzDiff": ("v0_collisionVzDiff", "V0J", 50, -1e-9, 1e-9, "Pos.Collision.vz minus Neg.Collision.vz (cm)", "V0s"),
}
# the profile of Pos.pt/pt against 1/pt, one curve per particle type: a 3-dimensional histogram and its projection
PROFILE = {"name": "hV0Profile", "variables": ["v0_invPt", "v0_posPtOverPt", "v0_pdg"], "nbins": [24, 60, 3],
           "range": [[0.0, 2.4], [0.0, 3.0], [-4000.0, 4000.0]], "source": "V0J", "axis": [1], "quantiles": [0.5],
           "unbinned_projections": True}


def page_tables(raw: dict) -> dict:
    """The ToyMC tables as page tables, one per role, with the role's prefix on every column. The V0s and cascades get
    the mass of each daughter track under the particle type of the row (the generator's own tables)."""
    out = {}
    for role, (table, prefix, columns) in ROLES.items():
        out[role] = {prefix + c: raw[table][c].to_numpy() for c in columns}
    v0_pdg, casc_pdg = raw["v0s"]["pdg"].to_numpy(), raw["cascades"]["pdg"].to_numpy()
    out["V0s"]["v0_massPos"] = np.array([MASS[V0_DAUGHTERS[int(p)][0]] for p in v0_pdg], dtype=np.float64)
    out["V0s"]["v0_massNeg"] = np.array([MASS[V0_DAUGHTERS[int(p)][1]] for p in v0_pdg], dtype=np.float64)
    out["Cascades"]["casc_massBachelor"] = np.array([MASS[CASCADE_DAUGHTERS[int(p)][1]] for p in casc_pdg], dtype=np.float64)
    return out


def page_description(raw: dict, how: str = "left") -> dict:
    """Everything bokehDrawSA.fromArray needs, as plain lists and dictionaries (the usual page description)."""
    tables = page_tables(raw)
    sources = [{"name": role, "data": tables[role]} for role in ROLES if role != MAIN]
    sources += [{"name": name, "left": None if child == MAIN else child, "right": parent, "left_on": [pointer],
                 "right_on": [], "how": how} for name, (child, parent, pointer) in JOINS.items()]
    sources += [{"name": name, "variables": [column], "nbins": nbins, "range": [low, high], "source": source}
                for name, (column, source, nbins, low, high, _, _) in HISTOGRAMS.items()]
    sources.append(dict(PROFILE))
    figures, index = [], {}

    def add(key, figure):
        index[key] = len(figures)
        figures.append(figure)

    for name, (_, source, _, _, _, title, counted) in HISTOGRAMS.items():
        add(name, [["bin_center"], ["bin_count"], {"source": name, "xAxisTitle": title, "yAxisTitle": counted,
                                                  "plotTitle": f"{title} [{'join' if source in JOINS else 'table'} {source}]"}])
    scatter = lambda key, x, y, xt, yt, title: add(key, [[x], [y], {"xAxisTitle": xt, "yAxisTitle": yt, "plotTitle": title}])
    scatter("sTrk", "TrkJ.trk_eta", "TrkJ.trk_pt", "track eta", "track pt (GeV/c)", "Tracks: pt against eta")
    scatter("sV0PosNeg", "V0J.neg_pt", "V0J.pos_pt", "Neg.pt (GeV/c)", "Pos.pt (GeV/c)", "V0s: Pos.pt against Neg.pt")
    scatter("sCascV0", "CascJ.v0_mass", "CascJ.casc_mass", "V0.mass (GeV/c2)", "cascade mass (GeV/c2)",
            "Cascades: mass against the mass of its V0")
    scatter("sV0Mass", "V0J.v0_mass", "V0J.v0_massFromTracks", "stored V0 mass (GeV/c2)", "mass from Pos and Neg (GeV/c2)",
            "V0s: the mass computed from the joined tracks against the stored mass")
    scatter("sCascMass", "CascJ.casc_mass", "CascJ.casc_massFromV0AndBachelor", "stored cascade mass (GeV/c2)",
            "mass from V0 and Bachelor (GeV/c2)", "Cascades: the mass computed from the joined V0 and track against the stored mass")
    add("pV0", [["bin_center_0"], ["mean"], {"source": PROFILE["name"] + "_1", "colorZvar": "bin_center_2",
                                             "errY": "std/sqrt(entries)", "xAxisTitle": "1/pt of the V0 (c/GeV)",
                                             "yAxisTitle": "mean of Pos.pt/pt",
                                             "plotTitle": "V0s: mean of Pos.pt/pt against 1/pt, one colour per particle type"}])
    add("sMain", [["casc_pt"], ["casc_mass"], {"xAxisTitle": "cascade pt (GeV/c)", "yAxisTitle": "cascade mass (GeV/c2)",
                                               "plotTitle": "Cascades: mass against pt (the main table, no join)"}])
    row = lambda *keys: [index[k] for k in keys]
    figure_layout = {
        "Collisions": [row("hCollVz", "hCollNTracks"), {"plot_height": 350}],
        "Tracks": [row("hTrkPt", "hTrkTgl", "hTrkCollVz"), row("sTrk"), {"plot_height": 300}],
        "V0s": [row("hV0Mass", "hV0DeltaMass", "hV0PosCollVz"), row("hV0PosPt", "hV0NegPt", "sV0PosNeg"), row("hV0Tgl", "pV0"),
                {"plot_height": 280}],
        "Cascades": [row("hCascMass", "hCascDeltaMass", "hCascPosCollVz"), row("hCascV0Mass", "hCascBachPt", "sCascV0"),
                     row("sMain"), {"plot_height": 280}],
        "Consistency": [row("hV0MassDiff", "sV0Mass", "hV0CollVzDiff"), row("hCascMassDiff", "sCascMass"),
                        {"plot_height": 320}],
    }
    return {"main": pd.DataFrame(tables[MAIN]), "tables": tables, "sources": sources, "aliases": [dict(a) for a in ALIASES],
            "figures": figures, "figure_index": index, "figure_layout": figure_layout}


def toy_mc_joins_dashboard(raw: dict, how: str = "left", n_point_render: int = 5000):
    """The page as a Bokeh layout, built by bokehDrawSA.fromArray; nothing is shown and nothing is written."""
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    d = page_description(raw, how)
    shown = sa.show
    sa.show = lambda *a, **k: None
    try:
        return sa.bokehDrawSA.fromArray(d["main"], None, d["figures"], [], sourceArray=d["sources"],
                                        aliasArray=d["aliases"], layout=d["figure_layout"], widgetLayout=[],
                                        nPointRender=n_point_render).pAll
    finally:
        sa.show = shown


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-collisions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--how", choices=("left", "inner"), default="left")
    parser.add_argument("--output", type=Path, required=True, help="the page file to write (.html)")
    args = parser.parse_args()
    from bokeh.io import save
    from bokeh.resources import INLINE
    raw = generate_toy_mc_events(n_collisions=args.n_collisions, seed=args.seed)
    save(toy_mc_joins_dashboard(raw, args.how), filename=str(args.output), resources=INLINE,
         title="ToyMC: collisions, tracks, V0s and cascades through joins")
    print("rows:", {name: len(table) for name, table in raw.items()})
    print("Wrote", args.output, "(%.1f MB)" % (args.output.stat().st_size / 1e6))


if __name__ == "__main__":
    main()
