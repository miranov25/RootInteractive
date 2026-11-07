"""Headless browser test engine for RootInteractive (PHASE_4).

Builds RootInteractive pages with the library's own builder, opens them in headless Chromium,
performs scripted steps and compares every result with an independent Python oracle.
Test infrastructure only: this package is excluded from the installed wheel (setup.py) and
changes no RootInteractive behaviour. Contract: PHASE_4_1_HeadlessBrowserTests_Proposal_v0_2.
"""
