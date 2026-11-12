"""The verdict on one scenario run, shared by the Node and the browser scenario tests (stage 4.2c-2b v0.2).

A run is accepted only if (a) every outcome is PASS or XFAIL (`result.ok`) and (b) every declared expected failure
really occurred, as XFAIL, at each of its steps. An expectation that never shows is a failure, not a silent success
(review of 4.2c-2b, GPT23: the Node scenario test checked only (a), so a run in which the B1 check never ran passed).
"""


def assert_as_declared(scenario, result) -> None:
    assert result.ok, result.text()
    for e in scenario.expect:
        for k in e.steps:
            got = [o.outcome for o in result.outcomes if o.check == e.check and o.step == k]
            assert got == ["XFAIL"], f"expected failure {e.bug} at step {k} on {e.check}: got {got}\n{result.text()}"
