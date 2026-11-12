# Pinned page scripts for the headless tests

Every RootInteractive page requests three scripts from the internet: pako 2.0.2 and vis 4.16.1 (from cdnjs, versioned) and
onnxruntime-web from jsDelivr **without a version**, so users' pages load whatever is newest at the time of viewing. The headless
engine serves pinned local copies instead, so a test never depends on the network or on an outside release; any other external
request is blocked and reported as an ERROR.

| Package | Version | Licence | Source |
|---|---|---|---|
| pako | 2.0.2 | MIT (`pako-2.0.2.LICENSE`) | npm registry tarball, integrity sha512 |
| vis | 4.16.1 | MIT or Apache-2.0 (`vis-4.16.1.LICENSE-MIT`, `vis-4.16.1.LICENSE-APACHE-2.0`) | npm registry tarball, integrity sha512 |
| onnxruntime-web | 1.30.0 | MIT (notice in the file header) | npm registry tarball, integrity sha512; the newest release older than two weeks on 2026-11-07 |

The files are **not committed** (git-ignored, about 1 MB). They are fetched and verified — tarball integrity and the sha256 of
every extracted file, both pinned in `engine/vendor.py` — with:

```
python -m RootInteractive.InteractiveDrawing.bokeh.headless.engine.vendor fetch
python -m RootInteractive.InteractiveDrawing.bokeh.headless.engine.vendor check
```

A CI container runs `fetch` while it is built, never during a job.
