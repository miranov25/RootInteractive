/*
 .L $RootInteractive/RootInteractive/Tools/treeInteractiveDraw.h+

*/
#ifndef treeInteractiveDraw_H
#define treeInteractiveDraw_H

#include "TPython.h"

namespace treeInteractiveDraw{
  const string importBokeh= R"(
from bokeh.io import output_notebook
from RootInteractive.Tools.aliTreePlayer import *
from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import *
from bokeh.io import curdoc
from RootInteractive.InteractiveDrawing.bokeh.bokehInteractiveParameters import *
from RootInteractive.Tools.compressArray import arrayCompressionRelative16, arrayCompressionRelative8
from RootInteractive.InteractiveDrawing.bokeh.palette import kBird256, kRainbow256
from RootInteractive.InteractiveDrawing.bokeh.bokehInteractiveTemplate import *
)";
  string makeTreeTemplateDraw(string treeName, string query,   string treeVar, string derVar, string deltaTemplate, string htmlOut, int verbose);
};

/*
    string treeName="tree"
    string query="abs(qPt)<10"
    string htmlOut="test.html"
    string treeVar=R"(["qPt","tgl","nclsF", "nCrossed", "clFraction", "isPrim0","nclsFExpqPt","qptNclFraction","tglNclFraction","fTPCNClsShared","fITSClusterMap","fTPCSignal","fTPCChi2NCl","fITSChi2NCl","trackVz"])";
    string derVar= R"(df["ITSOn"]=1*(df["fITSClusterMap"]>0); df["TPCOn"]=1*(df["nCrossed"]>0); df["isPrim0"]=1*(df["isPrim0"]>0) )"
    string deltaTemplate=""
    {deltaTemplate+=
        R"(widgetParams+=[["spinnerRange",["qPt"],{"name":"qPt"}], ["spinnerRange",["tgl"],{"name":"tgl"}], )"
        R"(["select",["isPrim0"],{"name":"isPrim0"}],["select",["ITSOn"],{"name":"ITSOn"}],   ["select",["TPCOn"],{"name":"TPCOn"}]])"
        //R"(["multiSelectBitmask",["isPrim0"],{"name":"isPrim0","how":"all","title":"isPrim"}],["multiSelectBitmask",["ITSOn"],{"name":"ITSOn","how":"all","title":"ITSOn"}],   ["multiSelectBitmask",["TPCOn"],{"name":"TPCOn","how":"all","title":"TPCOn"}]])"
        "\n"
        R"(widgetLayoutDesc["Select"]=[["qPt","tgl"],["isPrim0","ITSOn","TPCOn"]] )" "\n"
    ;
    }
    verbose=1;
    pythonCode= treeInteractiveDraw::makeTreeTemplateDraw(treeName,query,treeVar,derVar, deltaTemplate,htmlOut,verbose)
    //there are problems with the double in multiselect  (isPrim0) - code crashing with C++ stacktrace without proper python stacktrace
*/
///
/// \param treeName          -  input tree to query
/// \param query             -  tree selection as used in tree->Dra
/// \param treeVar           -  variables to export - * convention can be used
/// \param deltaTemplate     -  extension of the standard template
/// \param htmlOut           -  html output file
/// \param verbose           -  verbosity of the creation
/// \return                  - python code to generate the template dashboard
string treeInteractiveDraw::makeTreeTemplateDraw(string treeName, string query,   string treeVar, string derVar, string deltaTemplate, string htmlOut, int verbose){
    string pythonCode="\n";
    pythonCode+=treeInteractiveDraw::importBokeh;                    /// add import if needed
    //pythonCode+="tree=ROOT.gROOT.GetGlobal(\""+treeName+"\")\n";
    pythonCode+="output_file('"+htmlOut +"')\n";
    pythonCode+="df=tree2Panda(ROOT.tree,"+treeVar+",\""+query+"\")\n";
    pythonCode+=derVar+"\n";
    if (verbose>0) pythonCode+="print(df.head(5))\n";
    pythonCode+=R"(aliasArray, variables, parameterArray, widgetParams, widgetLayoutDesc, histoArray, figureArray, figureLayoutDesc = getDefaultVarsRatio(variables=df.columns.tolist())  )" "\n";
    pythonCode+=deltaTemplate.data();
    pythonCode+=R"(bokehDrawSA.fromArray(df, None, figureArray, widgetParams, layout=figureLayoutDesc, sizing_mode='scale_width', nPointRender=50000, widgetLayout=widgetLayoutDesc,)";
    pythonCode+=R"(parameterArray=parameterArray, histogramArray=histoArray, arrayCompression=arrayCompressionRelative16, rescaleColorMapper=True, aliasArray=aliasArray,palette=kRainbow256))";
    //TPython::Exec(pythonCode.data());
    return pythonCode;
}



#endif