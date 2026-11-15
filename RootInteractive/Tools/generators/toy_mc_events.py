"""PHASE_5_1: deterministic AO2D-like ToyMC for RootInteractive/ADF.

Four independent pandas tables: collisions, tracks, v0s, cascades.
Physics: on-shell isotropic two-body decays with exact Lorentz boosts, displaced
weak decays and configurable track-momentum smearing. Not an O2 detector or
reconstruction simulation. All foreign keys refer to dense ZERO-BASED row IDs.

Usage:
    from toy_mc_events import generate_toy_mc_events, make_alias_dataframes
    tables = generate_toy_mc_events(n_collisions=1000, seed=42)
    # adfs = make_alias_dataframes(tables)  # requires installed dfextensions

Run: python toy_mc_events.py --n-collisions 1000 --output-dir /tmp/toy_mc
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Masses in GeV/c^2; mean proper decay lengths c*tau in cm.
MASS = {
    211: 0.13957039, -211: 0.13957039,
    321: 0.493677, -321: 0.493677,
    2212: 0.93827208816, -2212: 0.93827208816,
    310: 0.497611, 3122: 1.115683, -3122: 1.115683,
    3312: 1.32171, -3312: 1.32171,
    3334: 1.67245, -3334: 1.67245,
}
CTAU_CM = {310: 2.6842, 3122: 7.89, -3122: 7.89,
           3312: 4.91, -3312: 4.91, 3334: 2.46, -3334: 2.46}
V0_DAUGHTERS = {
    310: (211, -211), 3122: (2212, -211), -3122: (211, -2212),
}
CASCADE_DAUGHTERS = {
    3312: (3122, -211), -3312: (-3122, 211),
    3334: (3122, -321), -3334: (-3122, 321),
}
CHARGE = {211: 1, -211: -1, 321: 1, -321: -1,
          2212: 1, -2212: -1}
TABLES = ("collisions", "tracks", "v0s", "cascades")

# Concise, portable ADF alias descriptions, separate from optional metadata.
ALIAS_EXPRESSIONS = {
    "collisions": {
        "isHighMultiplicity": "nTracks >= 12",
    },
    "tracks": {
        "ptFromPxPy": "sqrt(px**2 + py**2)",
        "tgl": "sinh(eta)",
        "signedInvPt": "charge / pt",
        "collisionVz": "Collision.vz",
    },
    "v0s": {
        "deltaMass": "mass - pdgMass",
        "massWindow20MeV": "abs(mass - pdgMass) < 0.020",
        "posPt": "Pos.pt",
        "negPt": "Neg.pt",
        "sameDaughterCollision": "Pos.collisionIdx == Neg.collisionIdx",
    },
    "cascades": {
        "deltaMass": "mass - pdgMass",
        "massWindow30MeV": "abs(mass - pdgMass) < 0.030",
        "v0Mass": "V0.mass",
        "bachelorPt": "Bachelor.pt",
        "v0Pt": "V0.pt",
    },
}
VARIABLE_METADATA = {
    "collisions": {
        "vz": {"unit": "cm", "title": "Collision z (cm)"},
        "nTracks": {"title": "Tracks per collision"},
        "isHighMultiplicity": {"title": "N(tracks) >= 12"},
    },
    "tracks": {
        "pt": {"unit": "GeV/c", "title": "Track pT (GeV/c)"},
        "eta": {"title": "Track pseudorapidity"},
        "ptFromPxPy": {"unit": "GeV/c", "title": "pT from px, py (GeV/c)"},
        "signedInvPt": {"unit": "c/GeV", "title": "q / pT (c/GeV)"},
        "collisionVz": {"unit": "cm", "title": "Associated collision z (cm)"},
    },
    "v0s": {
        "mass": {"unit": "GeV/c^2", "title": "V0 reconstructed mass (GeV/c^2)"},
        "deltaMass": {"unit": "GeV/c^2", "title": "V0 mass minus hypothesis (GeV/c^2)"},
        "decayRadius": {"unit": "cm", "title": "V0 displacement from production vertex (cm)"},
    },
    "cascades": {
        "mass": {"unit": "GeV/c^2", "title": "Cascade reconstructed mass (GeV/c^2)"},
        "deltaMass": {"unit": "GeV/c^2", "title": "Cascade mass minus hypothesis (GeV/c^2)"},
        "decayRadius": {"unit": "cm", "title": "Cascade displacement from PV (cm)"},
    },
}


def p4_from_pt_eta_phi(pt: float, eta: float, phi: float, mass: float) -> np.ndarray:
    """Return (energy, px, py, pz) in GeV; exact on-shell identity."""
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    energy = np.sqrt(mass * mass + px * px + py * py + pz * pz)
    return np.array([energy, px, py, pz], dtype=np.float64)


def inv_mass(p4: np.ndarray) -> float:
    return float(np.sqrt(max(0.0, p4[0] ** 2 - np.dot(p4[1:], p4[1:]))))


def lorentz_boost(p4: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Boost a 4-vector by parent velocity beta (c=1)."""
    b2 = float(np.dot(beta, beta))
    if not 0 <= b2 < 1:
        raise ValueError("Boost velocity must satisfy 0 <= beta^2 < 1")
    if b2 == 0:
        return p4.copy()
    gamma = 1.0 / np.sqrt(1.0 - b2)
    dot = float(np.dot(beta, p4[1:]))
    spatial = p4[1:] + ((gamma - 1.0) * dot / b2 + gamma * p4[0]) * beta
    return np.array([gamma * (p4[0] + dot), *spatial], dtype=np.float64)


def two_body_decay(rng: np.random.Generator, parent: np.ndarray,
                   mass1: float, mass2: float) -> tuple[np.ndarray, np.ndarray]:
    """Conserve mother 4-momentum exactly; isotropic decay in mother rest frame."""
    mass = inv_mass(parent)
    if mass < mass1 + mass2:
        raise ValueError("Parent mass below two-body threshold")
    lam = (mass * mass - (mass1 + mass2) ** 2) * (mass * mass - (mass1 - mass2) ** 2)
    pstar = np.sqrt(max(lam, 0.0)) / (2 * mass)
    ctheta = rng.uniform(-1.0, 1.0)
    phi = rng.uniform(-np.pi, np.pi)
    sth = np.sqrt(1.0 - ctheta * ctheta)
    p = pstar * np.array([sth * np.cos(phi), sth * np.sin(phi), ctheta])
    r1 = np.array([np.hypot(mass1, pstar), *p])
    r2 = np.array([np.hypot(mass2, pstar), *(-p)])
    beta = parent[1:] / parent[0]
    return lorentz_boost(r1, beta), lorentz_boost(r2, beta)


def _kinematics(p4: np.ndarray) -> dict:
    px, py, pz = (float(x) for x in p4[1:])
    pt = float(np.hypot(px, py))
    return {"pt": pt, "eta": float(np.arcsinh(pz / max(pt, 1e-15))),
            "phi": float(np.arctan2(py, px)), "px": px, "py": py, "pz": pz}


def _random_parent(rng: np.random.Generator, pdg: int) -> np.ndarray:
    pt = 0.22 + rng.gamma(2.0, 0.65)
    eta, phi = rng.uniform(-0.9, 0.9), rng.uniform(-np.pi, np.pi)
    return p4_from_pt_eta_phi(float(pt), float(eta), float(phi), MASS[pdg])


def _decay_vertex(rng: np.random.Generator, parent: np.ndarray,
                  pdg: int, start: np.ndarray) -> np.ndarray:
    p = float(np.linalg.norm(parent[1:]))
    if p == 0:
        return start.copy()
    # Lab mean decay length = c*tau * beta*gamma = c*tau * p/M.
    length = rng.exponential(CTAU_CM[pdg] * p / MASS[pdg])
    return start + length * parent[1:] / p


def _reco_track_p4(rng: np.random.Generator, truth: np.ndarray, pdg: int,
                   sigma_ptrel: float, sigma_ang: float) -> np.ndarray:
    kin = _kinematics(truth)
    # Log-normal pT factor avoids negative pT; mean adjusted to approximately one.
    noise = rng.normal(0.0, sigma_ptrel)
    pt = kin["pt"] * np.exp(noise - 0.5 * sigma_ptrel ** 2)
    return p4_from_pt_eta_phi(pt, kin["eta"] + rng.normal(0, sigma_ang),
                             kin["phi"] + rng.normal(0, sigma_ang), MASS[pdg])


class _EventBuilder:
    def __init__(self, rng: np.random.Generator, sigma_ptrel: float, sigma_ang: float):
        self.rng, self.sigma_ptrel, self.sigma_ang = rng, sigma_ptrel, sigma_ang
        self.collisions: list[dict] = []
        self.tracks: list[dict] = []
        self.v0s: list[dict] = []
        self.cascades: list[dict] = []
        self._v0_source_p4: list[np.ndarray] = []
        self._track_by_collision: list[list[int]] = []
        self._v0_by_collision: list[list[int]] = []

    def add_collision(self) -> int:
        idx = len(self.collisions)
        self.collisions.append({"rowId": idx, "vx": float(self.rng.normal(0, .005)),
                                "vy": float(self.rng.normal(0, .005)),
                                "vz": float(self.rng.normal(0, 5.0))})
        self._track_by_collision.append([])
        self._v0_by_collision.append([])
        return idx

    def pv(self, collision: int) -> np.ndarray:
        row = self.collisions[collision]
        return np.array([row["vx"], row["vy"], row["vz"]], dtype=np.float64)

    def add_track(self, collision: int, pdg: int, truth: np.ndarray,
                  vertex: np.ndarray, origin: str) -> int:
        reco = _reco_track_p4(self.rng, truth, pdg, self.sigma_ptrel, self.sigma_ang)
        idx = len(self.tracks)
        k = _kinematics(reco)
        tk = _kinematics(truth)
        self.tracks.append({"rowId": idx, "collisionIdx": collision,
                            "pdgTruth": pdg, "charge": CHARGE[pdg], "origin": origin,
                            **k, "truthPt": tk["pt"], "truthPx": tk["px"],
                            "truthPy": tk["py"], "truthPz": tk["pz"],
                            "truthE": float(truth[0]),
                            "prodX": float(vertex[0]), "prodY": float(vertex[1]),
                            "prodZ": float(vertex[2])})
        self._track_by_collision[collision].append(idx)
        return idx

    def track_p4(self, i: int, mass_hypothesis: float) -> np.ndarray:
        tr = self.tracks[i]
        return p4_from_pt_eta_phi(tr["pt"], tr["eta"], tr["phi"], mass_hypothesis)

    def add_v0(self, collision: int, pdg: int, pos: int, neg: int,
               vertex: np.ndarray, production_vertex: np.ndarray,
               is_signal: bool, origin: str, truth: np.ndarray | None = None) -> int:
        pdg_pos, pdg_neg = V0_DAUGHTERS[pdg]
        p4 = self.track_p4(pos, MASS[pdg_pos]) + self.track_p4(neg, MASS[pdg_neg])
        kin = _kinematics(p4)
        displacement = vertex - production_vertex
        pabs = float(np.linalg.norm(p4[1:]))
        dabs = float(np.linalg.norm(displacement))
        pointing = float(np.dot(displacement, p4[1:]) / (pabs * dabs)) if pabs * dabs else 1.0
        idx = len(self.v0s)
        self.v0s.append({"rowId": idx, "collisionIdx": collision,
                         "posTrackIdx": pos, "negTrackIdx": neg, "pdg": pdg,
                         "pdgMass": MASS[pdg], "mass": inv_mass(p4), "isSignal": bool(is_signal),
                         "origin": origin, "truthMass": inv_mass(truth) if truth is not None else np.nan,
                         "decayX": float(vertex[0]), "decayY": float(vertex[1]),
                         "decayZ": float(vertex[2]), "decayRadius": dabs,
                         "cosPointing": float(np.clip(pointing, -1.0, 1.0)), **kin})
        self._v0_source_p4.append(p4)
        self._v0_by_collision[collision].append(idx)
        return idx

    def spawn_v0(self, collision: int, pdg: int, parent: np.ndarray | None = None,
                 start: np.ndarray | None = None, origin: str = "standalone") -> int:
        if parent is None:
            parent = _random_parent(self.rng, pdg)
        if start is None:
            start = self.pv(collision)
        vertex = _decay_vertex(self.rng, parent, pdg, start)
        pos_pdg, neg_pdg = V0_DAUGHTERS[pdg]
        p4_pos, p4_neg = two_body_decay(self.rng, parent, MASS[pos_pdg], MASS[neg_pdg])
        pos = self.add_track(collision, pos_pdg, p4_pos, vertex, "v0_daughter")
        neg = self.add_track(collision, neg_pdg, p4_neg, vertex, "v0_daughter")
        return self.add_v0(collision, pdg, pos, neg, vertex, start, True, origin, parent)

    def add_cascade(self, collision: int, pdg: int, v0_idx: int, bachelor_idx: int,
                    vertex: np.ndarray, is_signal: bool, origin: str,
                    truth: np.ndarray | None = None) -> int:
        v0_p4 = self._v0_source_p4[v0_idx]
        bach_mass = MASS[CASCADE_DAUGHTERS[pdg][1]]
        reco = v0_p4 + self.track_p4(bachelor_idx, bach_mass)
        kin = _kinematics(reco)
        delta = vertex - self.pv(collision)
        pnorm = float(np.linalg.norm(reco[1:]))
        dnorm = float(np.linalg.norm(delta))
        cos_pa = float(np.dot(delta, reco[1:]) / (pnorm * dnorm)) if pnorm * dnorm else 1.0
        idx = len(self.cascades)
        self.cascades.append({"rowId": idx, "collisionIdx": collision,
                              "v0Idx": v0_idx, "bachelorTrackIdx": bachelor_idx,
                              "pdg": pdg, "pdgMass": MASS[pdg], "mass": inv_mass(reco),
                              "isSignal": bool(is_signal), "origin": origin,
                              "truthMass": inv_mass(truth) if truth is not None else np.nan,
                              "decayX": float(vertex[0]), "decayY": float(vertex[1]),
                              "decayZ": float(vertex[2]), "decayRadius": dnorm,
                              "cosPointing": float(np.clip(cos_pa, -1.0, 1.0)), **kin})
        return idx

    def spawn_cascade(self, collision: int, pdg: int) -> int:
        parent = _random_parent(self.rng, pdg)
        vertex = _decay_vertex(self.rng, parent, pdg, self.pv(collision))
        v0_pdg, bachelor_pdg = CASCADE_DAUGHTERS[pdg]
        v0_true, bachelor_true = two_body_decay(self.rng, parent, MASS[v0_pdg], MASS[bachelor_pdg])
        v0_idx = self.spawn_v0(collision, v0_pdg, v0_true, vertex, origin="cascade_daughter")
        bachelor_idx = self.add_track(collision, bachelor_pdg, bachelor_true, vertex, "cascade_bachelor")
        return self.add_cascade(collision, pdg, v0_idx, bachelor_idx, vertex,
                                True, "signal", truth=parent)

    def spawn_background_tracks(self, collision: int, count: int) -> None:
        species = (211, -211, 321, -321, 2212, -2212)
        for _ in range(count):
            pdg = int(self.rng.choice(species, p=[.35, .35, .09, .09, .06, .06]))
            p4 = _random_parent(self.rng, pdg)
            self.add_track(collision, pdg, p4, self.pv(collision), "primary_background")

    def spawn_background_v0s(self, collision: int, count: int) -> None:
        indices = self._track_by_collision[collision]
        pos = [i for i in indices if self.tracks[i]["charge"] > 0]
        neg = [i for i in indices if self.tracks[i]["charge"] < 0]
        if not pos or not neg:
            return
        for _ in range(count):
            a, b = int(self.rng.choice(pos)), int(self.rng.choice(neg))
            pdg = int(self.rng.choice((310, 3122, -3122)))
            vertex = self.pv(collision) + self.rng.normal(0, 1.5, 3)
            self.add_v0(collision, pdg, a, b, vertex, self.pv(collision),
                        False, "combinatorial")

    def spawn_background_cascades(self, collision: int, count: int) -> None:
        candidates = [v for v in self._v0_by_collision[collision]
                      if self.v0s[v]["pdg"] in (3122, -3122)]
        if not candidates:
            return
        for _ in range(count):
            v = int(self.rng.choice(candidates))
            is_lambda = self.v0s[v]["pdg"] == 3122
            charge = -1 if is_lambda else 1
            forbidden = {self.v0s[v]["posTrackIdx"], self.v0s[v]["negTrackIdx"]}
            tracks = [i for i in self._track_by_collision[collision]
                      if i not in forbidden and self.tracks[i]["charge"] == charge]
            if not tracks:
                continue
            bach = int(self.rng.choice(tracks))
            pdg = int(self.rng.choice((3312, 3334) if is_lambda else (-3312, -3334)))
            vertex = self.pv(collision) + self.rng.normal(0, 1.2, 3)
            self.add_cascade(collision, pdg, v, bach, vertex, False, "combinatorial")

    def finish(self) -> dict[str, pd.DataFrame]:
        count_tr = Counter(x["collisionIdx"] for x in self.tracks)
        count_v0 = Counter(x["collisionIdx"] for x in self.v0s)
        count_casc = Counter(x["collisionIdx"] for x in self.cascades)
        for row in self.collisions:
            c = row["rowId"]
            row.update(nTracks=count_tr[c], nV0=count_v0[c], nCascades=count_casc[c])
        tables = {"collisions": pd.DataFrame(self.collisions),
                  "tracks": pd.DataFrame(self.tracks),
                  "v0s": pd.DataFrame(self.v0s),
                  "cascades": pd.DataFrame(self.cascades)}
        # The schema stays stable even if a table is empty (e.g. one empty collision).
        # Use the first existing row as a dtype guide where possible; otherwise fixed dtypes.
        schema = {
            "tracks": {"rowId": "int32", "collisionIdx": "int32", "pdgTruth": "int32",
                       "charge": "int8", "origin": "object", "pt": "float64", "eta": "float64",
                       "phi": "float64", "px": "float64", "py": "float64", "pz": "float64",
                       "truthPt": "float64", "truthPx": "float64", "truthPy": "float64",
                       "truthPz": "float64", "truthE": "float64", "prodX": "float64",
                       "prodY": "float64", "prodZ": "float64"},
            "v0s": {"rowId": "int32", "collisionIdx": "int32", "posTrackIdx": "int32",
                    "negTrackIdx": "int32", "pdg": "int32", "pdgMass": "float64",
                    "mass": "float64", "isSignal": "bool", "origin": "object",
                    "truthMass": "float64", "decayX": "float64", "decayY": "float64",
                    "decayZ": "float64", "decayRadius": "float64", "cosPointing": "float64",
                    "pt": "float64", "eta": "float64", "phi": "float64",
                    "px": "float64", "py": "float64", "pz": "float64"},
            "cascades": {"rowId": "int32", "collisionIdx": "int32", "v0Idx": "int32",
                         "bachelorTrackIdx": "int32", "pdg": "int32", "pdgMass": "float64",
                         "mass": "float64", "isSignal": "bool", "origin": "object",
                         "truthMass": "float64", "decayX": "float64", "decayY": "float64",
                         "decayZ": "float64", "decayRadius": "float64", "cosPointing": "float64",
                         "pt": "float64", "eta": "float64", "phi": "float64",
                         "px": "float64", "py": "float64", "pz": "float64"},
        }
        for name in ("tracks", "v0s", "cascades"):
            df = tables[name]
            for col, dtype in schema[name].items():
                if col not in df:
                    df[col] = pd.Series(dtype=dtype)
                else:
                    df[col] = df[col].astype(dtype)
            tables[name] = df[list(schema[name])]
        tables["collisions"] = tables["collisions"].astype({"rowId": "int32",
                              "nTracks": "int32", "nV0": "int32", "nCascades": "int32"})
        return tables


def generate_toy_mc_events(n_collisions: int = 100, seed: int = 42,
                           mean_background_tracks: float = 8.0,
                           mean_standalone_v0: float = 0.9,
                           mean_cascades: float = 0.40,
                           mean_background_v0: float = 1.2,
                           mean_background_cascade: float = 0.45,
                           sigma_ptrel: float = 0.01, sigma_ang: float = 0.002,
                           reserve_empty_collision: bool = True,
                           inject_orphans: bool = False) -> dict[str, pd.DataFrame]:
    """Build exactly four AO2D-like tables with fully valid integer row references.

    First collisions include guaranteed K0s, Lambda, antiLambda, Xi± and Omega±
    where enough collisions are requested. The final collision is deliberately empty
    by default, to test zero-child inverse lookup. Local RNG; no global RNG reset.
    """
    if n_collisions < 1:
        raise ValueError("n_collisions must be >= 1")
    rates = (mean_background_tracks, mean_standalone_v0, mean_cascades,
             mean_background_v0, mean_background_cascade)
    if any(not np.isfinite(x) or x < 0 for x in rates):
        raise ValueError("Poisson mean rates must be finite and nonnegative")
    if not (0 <= sigma_ptrel <= .30 and 0 <= sigma_ang <= .30):
        raise ValueError("Resolutions must be in [0, 0.30]")
    rng = np.random.default_rng(seed)
    builder = _EventBuilder(rng, sigma_ptrel, sigma_ang)
    for _ in range(n_collisions):
        builder.add_collision()
    last_real = n_collisions - int(reserve_empty_collision and n_collisions > 1)
    # Explicit, reproducible non-vacuous physics cases (without relying on Poisson draws).
    compulsory_v0 = {0: (310, 3122), 1: (-3122,)}
    compulsory_cascades = {0: (3312,), 1: (-3312,), 2: (3334,), 3: (-3334,)}
    for c in range(last_real):
        builder.spawn_background_tracks(c, int(rng.poisson(mean_background_tracks)))
        for pdg in compulsory_v0.get(c, ()):
            builder.spawn_v0(c, pdg)
        for pdg in compulsory_cascades.get(c, ()):
            builder.spawn_cascade(c, pdg)
        for _ in range(int(rng.poisson(mean_standalone_v0))):
            builder.spawn_v0(c, int(rng.choice((310, 3122, -3122))))
        for _ in range(int(rng.poisson(mean_cascades))):
            builder.spawn_cascade(c, int(rng.choice((3312, -3312, 3334, -3334),
                                                  p=(.4, .4, .1, .1))))
        builder.spawn_background_v0s(c, int(rng.poisson(mean_background_v0)))
        builder.spawn_background_cascades(c, int(rng.poisson(mean_background_cascade)))
    tables = builder.finish()
    return inject_orphan_test_rows(tables) if inject_orphans else tables


def inject_orphan_test_rows(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Return separate stress tables with intentional -1 missing-parent references.

    Adds: one track without a collision; one V0 missing its positive daughter;
    three cascades (missing V0, missing bachelor, V0 with missing daughter).
    The normal physical fixture is not mutated and has no dangling references.
    """
    frames = {k: v.copy(deep=True) for k, v in tables.items()}
    tr, v0, ca, co = (frames[k] for k in ("tracks", "v0s", "cascades", "collisions"))
    if not len(tr) or not len(v0) or not len(ca):
        raise ValueError("Orphan stress mode needs at least one signal cascade and V0")
    c = int(v0.iloc[0].collisionIdx)
    tr_orphan = tr.iloc[0].copy()
    tr_orphan["rowId"] = len(tr)
    tr_orphan["collisionIdx"] = -1
    tr_orphan["origin"] = "orphan_test"
    frames["tracks"] = pd.concat([tr, pd.DataFrame([tr_orphan])], ignore_index=True).astype(tr.dtypes.to_dict())
    v0_orphan = v0.iloc[0].copy()
    v0_orphan["rowId"] = len(v0)
    v0_orphan["posTrackIdx"] = -1
    v0_orphan["origin"] = "orphan_test"
    v0_orphan["isSignal"] = False
    for k in ("mass", "truthMass", "pt", "px", "py", "pz", "eta", "phi", "cosPointing"):
        v0_orphan[k] = np.nan
    frames["v0s"] = pd.concat([v0, pd.DataFrame([v0_orphan])], ignore_index=True).astype(v0.dtypes.to_dict())
    base = ca.iloc[0].copy()
    rows = []
    for kind in ("missing_v0", "missing_bachelor", "missing_v0_daughter"):
        candidate = base.copy()
        candidate["rowId"] = len(ca) + len(rows)
        candidate["isSignal"] = False
        candidate["origin"] = kind
        candidate["truthMass"] = np.nan
        for k in ("mass", "pt", "px", "py", "pz", "eta", "phi", "cosPointing"):
            candidate[k] = np.nan
        if kind == "missing_v0":
            candidate["v0Idx"] = -1
        elif kind == "missing_bachelor":
            candidate["bachelorTrackIdx"] = -1
        else:
            candidate["v0Idx"] = len(v0)
        rows.append(candidate)
    frames["cascades"] = pd.concat([ca, pd.DataFrame(rows)], ignore_index=True).astype(ca.dtypes.to_dict())
    # Counts refer only to records whose own collisionIdx is valid.
    for child, name in (("v0s", "nV0"), ("cascades", "nCascades")):
        counts = np.bincount(frames[child]["collisionIdx"].to_numpy(), minlength=len(co))
        frames["collisions"][name] = counts.astype(np.int32)
    return frames


def validate_relations(tables: dict[str, pd.DataFrame],
                       allow_orphans: bool = False) -> dict[str, int]:
    """Raise if row-index or collision-consistency invariants fail."""
    if set(tables) != set(TABLES):
        raise ValueError("Expected exactly four named tables")
    for name in TABLES:
        df = tables[name]
        if not np.array_equal(df["rowId"].to_numpy(), np.arange(len(df))):
            raise ValueError(f"{name}: rowId is not the dense positional index")
    def index_ok(child, col, parent):
        x = tables[child][col].to_numpy(dtype=np.int64)
        invalid = (x < -1 if allow_orphans else x < 0) | (x >= len(tables[parent]))
        if np.any(invalid):
            raise ValueError(f"{child}.{col}: invalid {parent} reference")
    index_ok("tracks", "collisionIdx", "collisions")
    for col in ("posTrackIdx", "negTrackIdx"):
        index_ok("v0s", col, "tracks")
    index_ok("v0s", "collisionIdx", "collisions")
    index_ok("cascades", "v0Idx", "v0s")
    index_ok("cascades", "bachelorTrackIdx", "tracks")
    index_ok("cascades", "collisionIdx", "collisions")
    trc = tables["tracks"]["collisionIdx"].to_numpy()
    v0, ca = tables["v0s"], tables["cascades"]
    vc = v0["collisionIdx"].to_numpy()
    for field in ("posTrackIdx", "negTrackIdx"):
        idx = v0[field].to_numpy()
        valid = idx >= 0
        if np.any(trc[idx[valid]] != vc[valid]):
            raise ValueError("V0 daughters disagree with candidate collision")
    for field, parent_collisions in (("v0Idx", vc), ("bachelorTrackIdx", trc)):
        idx = ca[field].to_numpy()
        valid = idx >= 0
        if np.any(parent_collisions[idx[valid]] != ca["collisionIdx"].to_numpy()[valid]):
            raise ValueError("Cascade V0/bachelor disagree with candidate collision")
    co = tables["collisions"]
    for child, key in (("tracks", "nTracks"), ("v0s", "nV0"), ("cascades", "nCascades")):
        refs = tables[child]["collisionIdx"].to_numpy()
        counts = np.bincount(refs[refs >= 0], minlength=len(co))
        if not np.array_equal(counts, co[key].to_numpy()):
            raise ValueError(f"collisions.{key} count mismatch")
    return {name: len(tables[name]) for name in TABLES}


def independent_gather_oracle(tables: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    """Independent positional oracle. Missing index -1 yields NaN, not last row."""
    def take(values, indices):
        a = np.asarray(values, dtype=float)
        index = np.asarray(indices, dtype=np.int64)
        out = np.full(index.shape, np.nan)
        valid = (index >= 0) & (index < len(a))
        out[valid] = a[index[valid]]
        return out

    c, t, v, ca = (tables[n] for n in TABLES)
    ti = t["collisionIdx"].to_numpy()
    vp, vn = v["posTrackIdx"].to_numpy(), v["negTrackIdx"].to_numpy()
    cv, cb = ca["v0Idx"].to_numpy(), ca["bachelorTrackIdx"].to_numpy()
    return {
        "tracks.collisionVz": take(c["vz"], ti),
        "v0s.posPt": take(t["pt"], vp),
        "v0s.negPt": take(t["pt"], vn),
        "v0s.posCollisionIdx": take(ti, vp),
        "v0s.negCollisionIdx": take(ti, vn),
        "cascades.v0Mass": take(v["mass"], cv),
        "cascades.bachelorPt": take(t["pt"], cb),
        "cascades.collisionViaV0Pos": take(take(ti, vp), cv),
        "cascades.collisionViaV0Neg": take(take(ti, vn), cv),
        "cascades.collisionViaBachelor": take(ti, cb),
    }


def select_collision_descendants(tables: dict[str, pd.DataFrame],
                                 collision_ids: list[int]) -> dict[str, np.ndarray]:
    """Independent ALL-PATHS selection policy (valid candidates must agree).

    Returns positional indices, no duplicates and sorted in table order.
    """
    selected = np.unique(np.asarray(collision_ids, dtype=np.int64))
    if len(selected) and (np.any(selected < 0) or np.any(selected >= len(tables["collisions"]))):
        raise ValueError("Selected collision row index outside table")
    tr = tables["tracks"]
    v = tables["v0s"]
    ca = tables["cascades"]
    trmask = np.isin(tr["collisionIdx"].to_numpy(), selected)
    vmask = (np.isin(v["posTrackIdx"].to_numpy(), np.flatnonzero(trmask)) &
             np.isin(v["negTrackIdx"].to_numpy(), np.flatnonzero(trmask)) &
             np.isin(v["collisionIdx"].to_numpy(), selected))
    cmask = (np.isin(ca["v0Idx"].to_numpy(), np.flatnonzero(vmask)) &
             np.isin(ca["bachelorTrackIdx"].to_numpy(), np.flatnonzero(trmask)) &
             np.isin(ca["collisionIdx"].to_numpy(), selected))
    return {"collisions": selected.astype(np.int32),
            "tracks": np.flatnonzero(trmask).astype(np.int32),
            "v0s": np.flatnonzero(vmask).astype(np.int32),
            "cascades": np.flatnonzero(cmask).astype(np.int32)}


def make_alias_dataframes(tables: dict[str, pd.DataFrame],
                          attach_metadata: bool = True) -> dict:
    """Build four root ADFs, with independent ADF views for each relation path.

    ADF ruling D2 forbids one mutable ADF object under multiple names in one
    graph. Every view wraps a SHALLOW pandas copy: physical arrays are shared,
    while computed columns, aliases and metadata stay local to the view.
    """
    try:
        from dfextensions import AliasDataFrame
    except ImportError as exc:
        raise ImportError("make_alias_dataframes requires `dfextensions` on the ADF environment") from exc
    # D2 (2026-07-27): one mutable ADF object cannot be reached by two
    # names/paths in the same frame graph. Share pandas data, NOT ADF objects.
    # Every relation role therefore gets its own ADF view and descendants.
    # Shallow DataFrame copies isolate column mutations without copying data arrays.
    relations = {
        "tracks": (("Collision", "collisions", "collisionIdx"),),
        "v0s": (("Pos", "tracks", "posTrackIdx"),
                ("Neg", "tracks", "negTrackIdx")),
        "cascades": (("V0", "v0s", "v0Idx"),
                     ("Bachelor", "tracks", "bachelorTrackIdx")),
    }

    def make_view(table: str, path: str):
        view = AliasDataFrame(tables[table].copy(deep=False),
                              schema_id=f"RI_PHASE_5_1_{path}_v0_2")
        for role, child_table, foreign_key in relations.get(table, ()):
            child = make_view(child_table, f"{path}_{role}")
            view.register_subframe(role, child, index_columns=foreign_key,
                                   right_index_columns="rowId")
        for alias, expression in ALIAS_EXPRESSIONS[table].items():
            view.add_alias(alias, expression)
        if attach_metadata:
            view.set_columns_metadata(VARIABLE_METADATA.get(table, {}))
        return view

    return {name: make_view(name, name) for name in TABLES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-collisions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--inject-orphans", action="store_true",
                        help="Add a few -1 missing-parent references for join stress tests")
    args = parser.parse_args()
    tables = generate_toy_mc_events(n_collisions=args.n_collisions, seed=args.seed,
                                    inject_orphans=args.inject_orphans)
    print("PHASE_5_1 ToyMC tables:",
          validate_relations(tables, allow_orphans=args.inject_orphans))
    print("Physics truth closure (signal masses):")
    for name in ("v0s", "cascades"):
        df = tables[name]
        signal = df[df["isSignal"]]
        err = (signal["truthMass"] - signal["pdgMass"]).abs()
        print(f"  {name}: {len(signal)} signal candidates, max |Mtruth-Mpdg| = "
              f"{err.max() if len(err) else 0:.3e} GeV/c^2")
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            path = args.output_dir / f"toy_mc_{name}.csv"
            df.to_csv(path, index=False, float_format="%.17g")
            print("Wrote", path)


if __name__ == "__main__":
    main()
