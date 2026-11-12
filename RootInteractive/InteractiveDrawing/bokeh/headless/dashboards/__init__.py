"""PHASE_4_7: the oracle test dashboards - RootInteractive's own example dashboards (test_bokehClientHistogram.py) as
end-to-end browser tests, compared with an independent numpy oracle.

capture  - builds one original dashboard in a child process on the opt-in reproducible table and saves the real page
oracle   - the expected responses, computed with numpy from the raw table only
compare  - the fail-closed comparator shared by the tests and their falsifiers
video    - the narrated review video (optional; never needed by routine runs)
release  - the release check over the dashboards' health files, with the bugs accepted for release (waivers.json)
"""
