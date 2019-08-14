try:
    import ROOT
    ROOT.gSystem.Load("$ALICE_ROOT/lib/libSTAT.so")

except ImportError:
    pass