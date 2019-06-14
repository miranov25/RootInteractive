py.test --nbval $(ls */*/*ipynb | grep -v aux | grep -v TTreeHn)

# py.test --nbval tutorial/bokehDraw/bokehTreeDraw.ipynb    #OK
# py.test --nbval tutorial/bokehDraw/THnInteractive.ipynb   # test OK - but I had to remove tree.ls()                                                                                                              [ 48%]#
# py.test --nbval tutorial/bokehDraw/THnVisualization.ipynb .....            #test OK                                                                                                  [ 68%]
# py.test --nbval tutorial/MLpipeline/plotRegressionForestError.ipynb ....                                                                                                     [ 84%]
# py.test --nbval tutorial/QuantileRegression/QuantilRegressionForest.ipynb