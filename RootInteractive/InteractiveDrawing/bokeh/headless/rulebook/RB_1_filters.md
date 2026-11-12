# RB-1 — Filters (stage 4.2a entries)

Each rule: statement · source evidence at RootInteractive `4e8d783` · status. Only verified rules may be used by an oracle
(`oracles/filters.py`). Statuses: **verified in code**, **verified by measurement** (scenario + report), **to verify**, **known defect**.

| Rule | Statement | Evidence | Status |
|---|---|---|---|
| RB-1.1 | A range filter keeps a row when `low <= value <= high`, inclusive at both ends. | `RangeFilter.ts` 81 and 115: `(col[i] >= low) && (col[i] <= high)` | verified in code |
| RB-1.2 | A missing value (NaN) is never kept by a range filter, also at the widget's full bounds. | follows from RB-1.1 (NaN fails both comparisons); measured by scenario `selection/rb1_nan_excluded_at_full_bounds` (report in the stage-4.2a packet) | verified by measurement once that report is attached (erratum E-3) |
| RB-1.3 | The selection is the intersection of all active filters. | intended behaviour | known defect B1 (two filters excluding rows at load: only the first applies) |
| RB-1.4 | A range widget's default limits are `nanmin`/`nanmax` of the column **before** transport. | `bokehTools.py` 1510–1511 | verified in code; under lossy transport extreme rows drop at load (candidate B9, D-18) |
| RB-1.5 | Widget values used by the oracle are read from the page, never from the declaration. | engine rule (proposal §4, §6.3) | — |

Later entries (multi-select, missing-value option, 32-bit packing) arrive with the selection family in stage 4.4.
