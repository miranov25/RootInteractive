# RB-2 — Histogram bin membership (RI-HIST-BINEDGE-20261115-01)

**Candidate scientific-contract migration (GPT31.RI; final code review pending).**
This replaces the historical implementation-arithmetic rule, which allowed
wrong-side assignment at represented internal bin edges.

For finite `lo < hi` and positive integral `nbins`, define **one** binary64
edge vector `edges = numpy.linspace(lo, hi, nbins + 1)` with exactly matching
browser edge representation. The vector must be strictly increasing; the
numerical domain must be finite with representable, finite width.

- **RB-2.1 (1-D)** Each bin `i < nbins-1` is `[edges[i], edges[i+1])`.
- **RB-2.2 (N-D)** Apply the same membership independently on every axis,
  with first-variable-fastest flattened index. Any excluded axis drops the row.
- **RB-2.3 (upper endpoint)** The final bin is `[edges[nbins-1], edges[nbins]]`,
  including the exact represented `hi` value.
- **RB-2.4 (excluded values)** Values below `lo`, above `hi`, `NaN`, `+Inf`,
  and `-Inf` do not contribute to the histogram.
- **RB-2.5 (finite handling)** All axes must pass the finite-value test,
  including 1-D and N-D models (historical B8 fixed in PHASE_5 Batch 2A).
- **RB-2.6 (load selection)** The histogram respects the active selection
  when the page first loads (historical B2 fixed in PHASE_5 Batch 1).
- **RB-2.7 (exact internal edge)** At a represented internal edge `edges[i]`,
  the value belongs in right bin `i`. Its representable predecessor belongs
  in the left bin if in range; its successor belongs in the right bin. No ULP
  tolerance may move a value across a represented edge.
- **RB-2.8 (weighted/profile)** Weighted histograms sum weights in the exact
  assigned cells. Profiles and live selection/rebin updates must use the same
  membership. An aggregation-order tolerance is permitted only for floating
  sums, **not** for deciding bin membership.

**Independent reference:** `numpy.histogram` / `numpy.histogramdd` with the
*same represented edge vector*, or explicit comparisons with those edges.
The fast production index estimate may be used only after an edge-membership
correction. The old `(value * scale +/- origin) | 0` operation is not a
scientific oracle; it is retained only in labeled negative-control mutants.

**Historical bugs to keep detectable:**
- **B8:** 1-D NaN incorrectly entered bin 0; assert exclusion and reintroduce
  the defect by mutation.
- **B10:** upper-edge predecessor could overflow the final bin and be dropped
  or spill into an adjacent N-D row; negative control must fail numerically.
- **RI-HIST-BINEDGE-20261115-01:** interior represented-edge values assigned
  left by arithmetic rounding; strict edge/nextafter checks must catch it.

**Pending Architect policy (NOT silently resolved by this test migration):**
invalid/degenerate and single-point live ranges, overflow or non-increasing
binary64 edges, finite-width limits, cross-project edge transport, and
unrelated-dashboard survival under invalid updates. The present production
candidate can throw on those ranges. Neither this rulebook edit nor a green
histogram regression proves the invalid-live-range gate.
