"""PHASE_5_1 independent physics / relational tests; no RI product-code changes."""
from __future__ import annotations

import importlib.util
import sys
import types

import numpy as np
import pandas as pd
import pytest

from toy_mc_events import (
    ALIAS_EXPRESSIONS, CASCADE_DAUGHTERS, CHARGE, MASS, TABLES,
    VARIABLE_METADATA, generate_toy_mc_events, independent_gather_oracle,
    inv_mass, lorentz_boost, make_alias_dataframes, p4_from_pt_eta_phi,
    select_collision_descendants, two_body_decay, validate_relations,
)


@pytest.fixture(scope="module")
def tables():
    return generate_toy_mc_events(n_collisions=10, seed=42)


def test_lorentz_two_body_exact_conservation_and_on_shell():
    rng = np.random.default_rng(42)
    for species, (d1, d2) in CASCADE_DAUGHTERS.items():
        parent = p4_from_pt_eta_phi(1.8, .43, -.8, MASS[species])
        a, b = two_body_decay(rng, parent, MASS[d1], MASS[d2])
        np.testing.assert_allclose(a + b, parent, rtol=2e-13, atol=2e-13)
        assert abs(inv_mass(a) - MASS[d1]) < 5e-12
        assert abs(inv_mass(b) - MASS[d2]) < 5e-12
    resting = p4_from_pt_eta_phi(0., 0., 0., MASS[3312])
    np.testing.assert_array_equal(lorentz_boost(resting, np.zeros(3)), resting)
    with pytest.raises(ValueError, match="threshold"):
        two_body_decay(rng, resting, MASS[3122], MASS[2212])
    with pytest.raises(ValueError, match="velocity"):
        lorentz_boost(resting, np.array([1., 0., 0.]))


def test_four_tables_unique_keys_correct_types(tables):
    result = validate_relations(tables)
    assert tuple(tables) == TABLES
    assert all(result[name] > 0 for name in TABLES)
    assert len(tables["collisions"]) == 10
    for name, df in tables.items():
        assert df["rowId"].dtype == np.int32
        assert not df.isna().any().get("rowId", False)
        assert np.array_equal(df["rowId"], np.arange(len(df)))
    for name in ("tracks", "v0s", "cascades"):
        assert tables[name]["collisionIdx"].dtype == np.int32
    for name, keys in {"v0s": ("posTrackIdx", "negTrackIdx"),
                       "cascades": ("v0Idx", "bachelorTrackIdx")}.items():
        for k in keys:
            assert tables[name][k].dtype == np.int32
            assert (tables[name][k] >= 0).all()
    # Last collision has deliberately no children: non-vacuous reverse-join test.
    assert tables["collisions"].iloc[-1][["nTracks", "nV0", "nCascades"]].sum() == 0


def test_nonvacuous_species_backgrounds_and_branches(tables):
    v, ca, tr = (tables[x] for x in ("v0s", "cascades", "tracks"))
    assert {310, 3122, -3122} <= set(v[v.isSignal].pdg)
    assert {3312, -3312, 3334, -3334} <= set(ca[ca.isSignal].pdg)
    assert (~v.isSignal).sum() > 0
    assert (~ca.isSignal).sum() > 0
    assert any(tr.groupby("collisionIdx").size() >= 3)
    # Both V0 daughters and the cascade bachelor refer to distinct track rows.
    assert (v.posTrackIdx != v.negTrackIdx).all()
    for r in ca.itertuples():
        v0 = v.iloc[r.v0Idx]
        assert r.bachelorTrackIdx not in (v0.posTrackIdx, v0.negTrackIdx)
        assert CHARGE[CASCADE_DAUGHTERS[r.pdg][1]] == tr.iloc[r.bachelorTrackIdx].charge


def test_signal_truth_mass_closure_and_resolution(tables):
    for name in ("v0s", "cascades"):
        df = tables[name]
        sig = df[df.isSignal]
        np.testing.assert_allclose(sig.truthMass.to_numpy(), sig.pdgMass.to_numpy(),
                                   rtol=0, atol=8e-12)
        assert (sig.mass > 0).all()
        assert np.max(abs(sig.mass - sig.pdgMass)) < .20  # coarse sanity guard only
        assert np.max(abs(sig.mass - sig.pdgMass)) > 1e-6  # smearing changes mass
        assert df[~df.isSignal].truthMass.isna().all()
    # In zero-resolution mode every signal mass equals its PDG hypothesis.
    clean = generate_toy_mc_events(n_collisions=5, seed=1, sigma_ptrel=0, sigma_ang=0)
    for name in ("v0s", "cascades"):
        sig = clean[name][clean[name].isSignal]
        np.testing.assert_allclose(sig.mass, sig.pdgMass, rtol=0, atol=5e-12)


def test_per_event_count_and_exact_independent_gathers(tables):
    oracle = independent_gather_oracle(tables)
    t, v, c, ca = (tables[x] for x in ("tracks", "v0s", "collisions", "cascades"))
    assert len(oracle["v0s.posPt"]) == len(v)
    assert len(oracle["cascades.v0Mass"]) == len(ca)
    # Alternative truth via explicit pandas key merge, NOT original oracle's NumPy indexing.
    merged = t[["collisionIdx"]].merge(c[["rowId", "vz"]],
                                       left_on="collisionIdx", right_on="rowId",
                                       validate="many_to_one", sort=False)
    np.testing.assert_array_equal(oracle["tracks.collisionVz"], merged.vz.to_numpy())
    assert np.array_equal(oracle["v0s.posCollisionIdx"], v.collisionIdx.to_numpy())
    assert np.array_equal(oracle["v0s.negCollisionIdx"], v.collisionIdx.to_numpy())
    for k in ("cascades.collisionViaV0Pos", "cascades.collisionViaV0Neg",
              "cascades.collisionViaBachelor"):
        np.testing.assert_array_equal(oracle[k], ca.collisionIdx.to_numpy())


def test_selection_propagation_all_paths(tables):
    selected = select_collision_descendants(tables, [2, 0])
    assert np.array_equal(selected["collisions"], [0, 2])
    for name in ("tracks", "v0s", "cascades"):
        got = tables[name].iloc[selected[name]]
        assert got.collisionIdx.isin([0, 2]).all()
        assert len(got) == int(tables[name].collisionIdx.isin([0, 2]).sum())
    none = select_collision_descendants(tables, [])
    assert all(not len(arr) for arr in none.values())
    empty = select_collision_descendants(tables, [9])
    assert len(empty["collisions"]) == 1
    assert all(not len(empty[x]) for x in ("tracks", "v0s", "cascades"))
    with pytest.raises(ValueError, match="outside"):
        select_collision_descendants(tables, [999])


def test_bad_reference_and_cross_collision_mutations_refused(tables):
    wrong = dict(tables)
    wrong["v0s"] = tables["v0s"].copy()
    wrong["v0s"].loc[0, "posTrackIdx"] = 10 ** 7
    with pytest.raises(ValueError, match="invalid tracks reference"):
        validate_relations(wrong)

    wrong = dict(tables)
    wrong["cascades"] = tables["cascades"].copy()
    wrong["cascades"].loc[0, "v0Idx"] = -1
    with pytest.raises(ValueError, match="invalid v0s reference"):
        validate_relations(wrong)

    wrong = dict(tables)
    wrong["v0s"] = tables["v0s"].copy()
    first_collision = int(wrong["v0s"].iloc[0].collisionIdx)
    alt = tables["tracks"].index[tables["tracks"].collisionIdx != first_collision][0]
    wrong["v0s"].loc[0, "negTrackIdx"] = int(alt)
    with pytest.raises(ValueError, match="daughters disagree"):
        validate_relations(wrong)


def test_same_seed_byte_identical_and_separate_seed_changes():
    a = generate_toy_mc_events(n_collisions=12, seed=123)
    b = generate_toy_mc_events(n_collisions=12, seed=123)
    for name in TABLES:
        pd.testing.assert_frame_equal(a[name], b[name], check_exact=True)
    c = generate_toy_mc_events(n_collisions=12, seed=124)
    assert not a["collisions"].vz.equals(c["collisions"].vz)


def test_one_collision_and_rejected_parameters():
    tables = generate_toy_mc_events(n_collisions=1, seed=42,
                                    mean_background_tracks=0,
                                    mean_standalone_v0=0, mean_cascades=0,
                                    mean_background_v0=0, mean_background_cascade=0)
    validate_relations(tables)
    assert all(len(tables[x]) > 0 for x in ("v0s", "cascades"))  # forced coverage
    for kwargs in ({"n_collisions": 0}, {"mean_background_tracks": -1},
                   {"sigma_ang": -0.01}, {"sigma_ptrel": .35}):
        with pytest.raises(ValueError):
            generate_toy_mc_events(**kwargs)


def test_alias_expressions_metadata_catalog():
    assert set(ALIAS_EXPRESSIONS) == set(TABLES)
    assert "Collision.vz" == ALIAS_EXPRESSIONS["tracks"]["collisionVz"]
    assert "Pos.pt" == ALIAS_EXPRESSIONS["v0s"]["posPt"]
    assert "V0.mass" == ALIAS_EXPRESSIONS["cascades"]["v0Mass"]
    assert VARIABLE_METADATA["v0s"]["mass"]["unit"] == "GeV/c^2"


def test_adf_registration_contract_via_mock(monkeypatch, tables):
    """Enforce ADF D2: no mutable ADF object may occur at two graph paths."""
    class FakeADF:
        def __init__(self, df, schema_id=None):
            self.df, self.schema_id = df, schema_id
            self.subframes, self.aliases, self.metadata = {}, {}, {}

        def walk(self):
            yield self
            for child, _, _ in self.subframes.values():
                yield from child.walk()

        def register_subframe(self, name, adf, index_columns, right_index_columns=None):
            seen = {id(v) for v in self.walk()}
            if any(id(v) in seen for v in adf.walk()):
                raise ValueError("D2: duplicate AliasDataFrame object in frame graph")
            self.subframes[name] = (adf, index_columns, right_index_columns)

        def add_alias(self, name, expression):
            self.aliases[name] = expression

        def set_columns_metadata(self, metadata):
            self.metadata = metadata

    monkeypatch.setitem(sys.modules, "dfextensions", types.SimpleNamespace(AliasDataFrame=FakeADF))
    # Negative control: the actual error in v0.1 must be caught by this mock.
    root, shared = FakeADF(tables["v0s"]), FakeADF(tables["tracks"])
    root.register_subframe("Pos", shared, "posTrackIdx", "rowId")
    with pytest.raises(ValueError, match="D2"):
        root.register_subframe("Neg", shared, "negTrackIdx", "rowId")

    adfs = make_alias_dataframes(tables)
    assert len(adfs) == 4
    assert adfs["tracks"].subframes["Collision"][1:] == ("collisionIdx", "rowId")
    assert adfs["v0s"].subframes["Pos"][1:] == ("posTrackIdx", "rowId")
    assert adfs["v0s"].subframes["Neg"][1:] == ("negTrackIdx", "rowId")
    assert adfs["cascades"].subframes["V0"][1:] == ("v0Idx", "rowId")
    assert adfs["cascades"].subframes["Bachelor"][1:] == ("bachelorTrackIdx", "rowId")
    assert adfs["v0s"].aliases == ALIAS_EXPRESSIONS["v0s"]
    assert adfs["v0s"].metadata == VARIABLE_METADATA["v0s"]
    for name in TABLES:
        views = list(adfs[name].walk())
        assert len(views) == len({id(v) for v in views}), f"D2 duplicate in {name}"
    pos = adfs["v0s"].subframes["Pos"][0]
    neg = adfs["v0s"].subframes["Neg"][0]
    assert pos is not neg and pos.df is not neg.df
    assert pos.df is not tables["tracks"] and neg.df is not tables["tracks"]
    pd.testing.assert_frame_equal(pos.df, tables["tracks"])
    pd.testing.assert_frame_equal(neg.df, tables["tracks"])
    assert pos.subframes["Collision"][0] is not neg.subframes["Collision"][0]
    v0_view = adfs["cascades"].subframes["V0"][0]
    bachelor = adfs["cascades"].subframes["Bachelor"][0]
    assert v0_view is not adfs["v0s"] and bachelor is not adfs["tracks"]
    assert v0_view.subframes["Pos"][0] is not pos
    pd.testing.assert_frame_equal(bachelor.df, tables["tracks"])
    pd.testing.assert_frame_equal(v0_view.df, tables["v0s"])


def test_real_adf_if_installed(tables):
    if importlib.util.find_spec("dfextensions") is None:
        pytest.skip("dfextensions absent here; run real ADF smoke test on alma2")
    original = {name: tables[name].copy(deep=True) for name in TABLES}
    adfs = make_alias_dataframes(tables)
    oracle = independent_gather_oracle(tables)
    for table, alias in (("tracks", "collisionVz"), ("v0s", "posPt"),
                         ("v0s", "negPt"), ("cascades", "v0Mass"),
                         ("cascades", "bachelorPt")):
        # ADF __getitem__ reads physical columns, while eval resolves lazy aliases.
        np.testing.assert_allclose(np.asarray(adfs[table].eval(alias), dtype=float),
                                   oracle[f"{table}.{alias}"], rtol=0, atol=1e-12)
    ca = tables["cascades"]
    v0 = tables["v0s"]
    tr = tables["tracks"]
    pos_rows = v0.posTrackIdx.to_numpy()[ca.v0Idx.to_numpy()]
    expected_pos_pt = tr.pt.to_numpy()[pos_rows]
    np.testing.assert_allclose(np.asarray(adfs["cascades"].eval("V0.Pos.pt")),
                               expected_pos_pt, rtol=0, atol=1e-12)
    expected_vz = tables["collisions"].vz.to_numpy()[ca.collisionIdx.to_numpy()]
    for expr in ("V0.Pos.Collision.vz", "Bachelor.Collision.vz"):
        np.testing.assert_allclose(np.asarray(adfs["cascades"].eval(expr)),
                                   expected_vz, rtol=0, atol=1e-12)
    for name in TABLES:
        pd.testing.assert_frame_equal(tables[name], original[name], check_exact=True)


def test_independent_mass_reconstruction_from_referenced_tracks(tables):
    """External algebraic oracle: direct final-state E,p closure, without ToyMC inv_mass()."""
    tr, v, ca = (tables[k] for k in ("tracks", "v0s", "cascades"))

    def p4_track(i, mass):
        row = tr.iloc[int(i)]
        px, py, pz = (float(row[k]) for k in ("px", "py", "pz"))
        energy = np.sqrt(mass**2 + px**2 + py**2 + pz**2)
        return np.array([energy, px, py, pz])

    for row in v.itertuples():
        from toy_mc_events import V0_DAUGHTERS
        pdgp, pdgn = V0_DAUGHTERS[row.pdg]
        p4 = p4_track(row.posTrackIdx, MASS[pdgp]) + p4_track(row.negTrackIdx, MASS[pdgn])
        m = np.sqrt(max(0, p4[0]**2 - sum(x*x for x in p4[1:])))
        assert abs(m - row.mass) < 2e-12

    for row in ca.itertuples():
        from toy_mc_events import V0_DAUGHTERS
        vrow = v.iloc[row.v0Idx]
        pdgp, pdgn = V0_DAUGHTERS[vrow.pdg]
        _, bachelor_pdg = CASCADE_DAUGHTERS[row.pdg]
        p4 = (p4_track(vrow.posTrackIdx, MASS[pdgp]) +
              p4_track(vrow.negTrackIdx, MASS[pdgn]) +
              p4_track(row.bachelorTrackIdx, MASS[bachelor_pdg]))
        m = np.sqrt(max(0, p4[0]**2 - sum(x*x for x in p4[1:])))
        assert abs(m - row.mass) < 2e-12


def test_orphan_mode_explicit_nan_fill_and_fail_closed():
    original = generate_toy_mc_events(n_collisions=6, seed=43)
    stress = generate_toy_mc_events(n_collisions=6, seed=43, inject_orphans=True)
    for name in TABLES:
        before = original[name]
        after = stress[name].iloc[:len(before)].reset_index(drop=True)
        # Appending orphan candidate rows updates collision multiplicity counts.
        if name == "collisions":
            after = after.drop(columns=["nV0", "nCascades"])
            before = before.drop(columns=["nV0", "nCascades"])
        pd.testing.assert_frame_equal(after, before, check_exact=True)
    assert len(stress["tracks"]) == len(original["tracks"]) + 1
    assert len(stress["v0s"]) == len(original["v0s"]) + 1
    assert len(stress["cascades"]) == len(original["cascades"]) + 3
    # Counts are expected to increase for the appended V0/cascades; check separately.
    assert validate_relations(stress, allow_orphans=True)["cascades"] == len(stress["cascades"])
    with pytest.raises(ValueError, match="invalid collisions reference"):
        validate_relations(stress)
    oracle = independent_gather_oracle(stress)
    assert np.isnan(oracle["tracks.collisionVz"][-1])
    assert np.isnan(oracle["v0s.posPt"][-1])
    assert np.isnan(oracle["cascades.v0Mass"][-3])
    assert np.isnan(oracle["cascades.bachelorPt"][-2])
    assert np.isnan(oracle["cascades.collisionViaV0Pos"][-1])
    selection = select_collision_descendants(stress, [0])
    assert (len(stress["v0s"]) - 1) not in selection["v0s"]
    for row in range(len(stress["cascades"]) - 3, len(stress["cascades"])):
        assert row not in selection["cascades"]


# v0.4 scientific acceptance oracles. This literal table is deliberately independent
# of the generator's V0_DAUGHTERS, CASCADE_DAUGHTERS and CHARGE dictionaries.
# PDG codes: pion 211, kaon 321, proton 2212, K0S 310, Lambda 3122,
# Xi 3312, Omega 3334; signs distinguish charge conjugates.
_LITERAL_V0 = {310: (211, -211), 3122: (2212, -211), -3122: (211, -2212)}
_LITERAL_CASCADE = {3312: (3122, -211), -3312: (-3122, 211),
                    3334: (3122, -321), -3334: (-3122, 321)}
_LITERAL_CHARGE = {211: +1, -211: -1, 321: +1, -321: -1,
                   2212: +1, -2212: -1}


def _assert_literal_decay_channels(dataset):
    """T6/T7: signal PDG and signed daughter roles, independent of production maps."""
    tracks, v0s, cascades = (dataset[n] for n in ('tracks', 'v0s', 'cascades'))
    assert set(v0s.loc[v0s.isSignal, 'pdg']) == set(_LITERAL_V0)
    assert set(cascades.loc[cascades.isSignal, 'pdg']) == set(_LITERAL_CASCADE)
    for row in tracks.itertuples():
        assert row.charge == _LITERAL_CHARGE[row.pdgTruth], ('track charge', row.rowId)
    for row in v0s[v0s.isSignal].itertuples():
        pos, neg = _LITERAL_V0[row.pdg]
        got_pos, got_neg = tracks.iloc[row.posTrackIdx], tracks.iloc[row.negTrackIdx]
        assert (int(got_pos.pdgTruth), int(got_neg.pdgTruth)) == (pos, neg), ('V0 channel', row.rowId)
        assert (int(got_pos.charge), int(got_neg.charge)) == (+1, -1), ('V0 roles', row.rowId)
    for row in cascades[cascades.isSignal].itertuples():
        v0_pdg, bach_pdg = _LITERAL_CASCADE[row.pdg]
        assert int(v0s.iloc[row.v0Idx].pdg) == v0_pdg, ('cascade V0 identity', row.rowId)
        bach = tracks.iloc[row.bachelorTrackIdx]
        assert int(bach.pdgTruth) == bach_pdg, ('cascade bachelor PDG', row.rowId)
        assert int(bach.charge) == _LITERAL_CHARGE[bach_pdg], ('bachelor charge', row.rowId)


def test_literal_decay_channels_and_track_charges(tables):
    _assert_literal_decay_channels(tables)


def test_channel_and_charge_mutants_are_rejected(tables, monkeypatch):
    """T6/T7: mutate genuine generator channel tables, not the oracle's table."""
    import toy_mc_events as product
    with monkeypatch.context() as mp:
        mp.setitem(product.CASCADE_DAUGHTERS, 3334, (3122, -211))  # Omega bachelor wrongly pion
        bad = product.generate_toy_mc_events(n_collisions=10, seed=42)
        with pytest.raises(AssertionError, match='cascade bachelor PDG'):
            _assert_literal_decay_channels(bad)
    with monkeypatch.context() as mp:
        mp.setitem(product.V0_DAUGHTERS, -3122, (-2212, 211))  # antiparticle role reversed
        bad = product.generate_toy_mc_events(n_collisions=10, seed=42)
        with pytest.raises(AssertionError, match='V0 channel|V0 roles'):
            _assert_literal_decay_channels(bad)
    wrong_charge = {**tables, 'tracks': tables['tracks'].copy(deep=True)}
    first_pos = int(tables['v0s'].iloc[0].posTrackIdx)
    wrong_charge['tracks'].loc[first_pos, 'charge'] = -1
    with pytest.raises(AssertionError, match='track charge'):
        _assert_literal_decay_channels(wrong_charge)
    swapped = {**tables, 'v0s': tables['v0s'].copy(deep=True)}
    first_ks = int(swapped['v0s'].index[(swapped['v0s'].pdg == 310) & swapped['v0s'].isSignal][0])
    a, b = (int(swapped['v0s'].loc[first_ks, k]) for k in ('posTrackIdx', 'negTrackIdx'))
    swapped['v0s'].loc[first_ks, ['posTrackIdx', 'negTrackIdx']] = [b, a]
    with pytest.raises(AssertionError, match='V0 channel|V0 roles'):
        _assert_literal_decay_channels(swapped)


def _assert_proper_decay_length(decay_func):
    """T8: same uniform variate => length grows linearly with |p|/mass."""
    import toy_mc_events as product
    start = np.array([0.2, -0.7, 2.1], dtype=float)
    species = 3122
    momentum = np.array([1.2, 0.5, 2.1], dtype=float)
    values = []
    for factor in (0.4, 2.3):
        p = factor * momentum
        mother = np.array([np.hypot(product.MASS[species], np.linalg.norm(p)), *p])
        got = decay_func(np.random.default_rng(887), mother, species, start)
        length = float(np.linalg.norm(got - start))
        expected = (np.random.default_rng(887).exponential(1.0) *
                    7.89 * float(np.linalg.norm(p)) / 1.115683)
        np.testing.assert_allclose(length, expected, rtol=2e-13, atol=1e-12)
        np.testing.assert_allclose((got - start) / length, p / np.linalg.norm(p), atol=2e-13)
        values.append(length)
    np.testing.assert_allclose(values[1] / values[0], 2.3 / 0.4, rtol=2e-13)


def test_decay_vertex_independent_proper_lifetime_and_inverse_mutant():
    """T8: fixed external proper-length scale, two momenta, explicit inverse mutant."""
    import toy_mc_events as product
    _assert_proper_decay_length(product._decay_vertex)

    def inverse_mutant(rng, parent, pdg, start):
        p = float(np.linalg.norm(parent[1:]))
        scale = 7.89 * 1.115683 / p  # deliberately wrong M/|p|
        return start + rng.exponential(scale) * parent[1:] / p

    with pytest.raises(AssertionError):
        _assert_proper_decay_length(inverse_mutant)


def _assert_signal_production_vertices(dataset):
    """T9: check every signal daughter at the actual referenced decay vertex."""
    tr, v0s, cascades = (dataset[n] for n in ('tracks', 'v0s', 'cascades'))
    n_daughters = n_bachelor = n_downstream = 0
    for v in v0s[v0s.isSignal].itertuples():
        vertex = np.array([v.decayX, v.decayY, v.decayZ])
        for index in (v.posTrackIdx, v.negTrackIdx):
            daughter = tr.iloc[int(index)]
            actual = daughter[['prodX', 'prodY', 'prodZ']].to_numpy(dtype=float)
            np.testing.assert_allclose(actual, vertex, rtol=0, atol=1e-12)
            n_daughters += 1
    for c in cascades[cascades.isSignal].itertuples():
        vertex = np.array([c.decayX, c.decayY, c.decayZ])
        bachelor = tr.iloc[int(c.bachelorTrackIdx)]
        actual = bachelor[['prodX', 'prodY', 'prodZ']].to_numpy(dtype=float)
        np.testing.assert_allclose(actual, vertex, rtol=0, atol=1e-12)
        n_bachelor += 1
        v0 = v0s.iloc[int(c.v0Idx)]
        flight = np.array([v0.decayX, v0.decayY, v0.decayZ]) - vertex
        flight_norm = np.linalg.norm(flight)
        assert np.isfinite(flight).all() and flight_norm > 1e-9, ("V0 downstream distance", c.rowId)
        # Independent four-table oracle: use daughter truth momenta, not the stored V0 momentum.
        v0_truth_p = sum((tr.iloc[int(idx)][["truthPx", "truthPy", "truthPz"]]
                          .to_numpy(dtype=float) for idx in (v0.posTrackIdx, v0.negTrackIdx)),
                         np.zeros(3, dtype=float))
        momentum_norm = np.linalg.norm(v0_truth_p)
        assert np.isfinite(v0_truth_p).all() and momentum_norm > 0, ("V0 truth momentum", c.rowId)
        pointing_cosine = float(np.dot(flight, v0_truth_p) / (flight_norm * momentum_norm))
        assert pointing_cosine > 1 - 1e-9, ("V0 flies downstream along its momentum", c.rowId, pointing_cosine)
        n_downstream += 1
    assert n_daughters > 0 and n_bachelor > 0 and n_downstream > 0


def test_signal_production_vertices_and_coordinate_mutant(tables):
    _assert_signal_production_vertices(tables)
    mutant = {**tables, 'tracks': tables['tracks'].copy(deep=True)}
    mutant['tracks'].loc[:, 'prodX'] = 0.0  # independent forced-zero X mutant
    with pytest.raises(AssertionError):
        _assert_signal_production_vertices(mutant)


def test_signal_cascade_reflected_upstream_vertex_mutant(tables):
    """A backward V0 flight must fail despite self-consistent daughter production coordinates."""
    _assert_signal_production_vertices(tables)  # healthy reference must pass first
    mutant = {**tables, "v0s": tables["v0s"].copy(deep=True),
              "tracks": tables["tracks"].copy(deep=True)}
    signals = tables["cascades"][tables["cascades"].isSignal]
    assert len(signals) > 0
    cascade = signals.iloc[0]
    v0_idx = int(cascade.v0Idx)
    original_v0 = tables["v0s"].iloc[v0_idx]
    cascade_vertex = cascade[["decayX", "decayY", "decayZ"]].to_numpy(dtype=float)
    original_vertex = original_v0[["decayX", "decayY", "decayZ"]].to_numpy(dtype=float)
    reflected_vertex = 2 * cascade_vertex - original_vertex
    assert np.linalg.norm(reflected_vertex - original_vertex) > 1e-9
    vertex_columns = ["decayX", "decayY", "decayZ"]
    mutant["v0s"].loc[v0_idx, vertex_columns] = reflected_vertex
    for track_idx in (int(original_v0.posTrackIdx), int(original_v0.negTrackIdx)):
        mutant["tracks"].loc[track_idx, ["prodX", "prodY", "prodZ"]] = reflected_vertex
    # The original equality checks still pass; only forward flight can detect this mutation.
    with pytest.raises(AssertionError, match="downstream along its momentum"):
        _assert_signal_production_vertices(mutant)
