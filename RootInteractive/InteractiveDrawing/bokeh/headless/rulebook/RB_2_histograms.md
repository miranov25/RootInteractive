# RB-2 — Histograms (stage 4.2a entries)

Statement · source evidence at RootInteractive `4e8d783` · status. Implemented by `oracles/histograms.py`, which reproduces
RootInteractive's own arithmetic and never calls `numpy.histogram` (the two differ at bin edges).

| Rule | Statement | Evidence | Status |
|---|---|---|---|
| RB-2.1 | 1-D bin index: `scale = nbins/(max−min)`, `origin = −min·scale`, `bin = (v·scale + origin) \| 0` | `HistogramCDS.ts` 143–144, 227 | verified in code |
| RB-2.2 | N-D bin index per axis: `scale = nbins/(max−min)`, `origin = min·scale`, `bin = (v·scale − origin) \| 0`; flat index with strides, first variable fastest. In IEEE arithmetic `a + (−b) = a − b`, so 1-D and N-D agree | `HistoNdCDS.ts` 167–168, 330–345 | verified in code (was "to verify" in proposal v0.2) |
| RB-2.3 | The last edge is inclusive: `v == max` goes into the last bin | `HistogramCDS.ts` 225; `HistoNdCDS.ts` 341 | verified in code |
| RB-2.4 | Values outside `[min, max]` are **not counted** (there is no under- or overflow bin in `bin_count`); ±Inf therefore not counted | 1-D: `getbin` 222–223 returns −1 / `nbins`, counting keeps `0 <= bin < nbins` (174, 181, 189); N-D: 337 | verified in code |
| RB-2.5 | Non-finite values are not counted | N-D: `HistoNdCDS.ts` 337 `!isFinite(val)` | verified in code (N-D); 1-D: known defect B8 (NaN counted in bin 0) |
| RB-2.6 | The histogram follows the selection at page load | N-D: `HistoNdCDS.ts` 62 calls `change_selection()` | verified in code (N-D); 1-D: known defect B2 |
| RB-2.7 | Example of the edge difference: 0.3, 0.6, 0.7 on [0, 1] × 10 → RootInteractive bins 3, 6, 7; `numpy.histogram` 2, 5, 6 | unit test `test_rb2_bin_formula_is_rootinteractive_not_numpy` | verified by test |

Weights: a weighted histogram sums the row's weight instead of 1 (`HistoNdCDS.ts` 237–247); compared with a relative tolerance
of 1e-12, because the order of the floating sums is not part of the rule.
