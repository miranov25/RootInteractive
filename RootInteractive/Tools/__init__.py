try:
    import ROOT
    ROOT.gROOT.ProcessLine( "gErrorIgnoreLevel = 0xFFF;" ) # TODO - this is bad style - we should load library only when needed
    ROOT.gSystem.Load("$ALICE_ROOT/lib/libSTAT.so")

except ImportError:
    pass