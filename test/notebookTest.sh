#clean_ipynb */*/*ipynb
for note in $(ls */*/*ipynb); do
    nb-clean clean < ${note}  > ${note}2
    echo mv ${note}2 ${note}
    mv ${note}2 ${note}
done

py.test --nbval  $(ls */*/*ipynb | grep -v aux | grep -v TTreeHn) | tee pytest.log

# py.test --nbval tutorial/bokehDraw/bokehTreeDraw.ipynb    #OK
# py.test --nbval tutorial/bokehDraw/THnInteractive.ipynb   # test OK - but I had to remove tree.ls()                                                                                                              [ 48%]#
# py.test --nbval tutorial/bokehDraw/THnVisualization.ipynb .....            #test OK                                                                                                  [ 68%]
# py.test --nbval tutorial/MLpipeline/plotRegressionForestError.ipynb ....                                                                                                     [ 84%]
# py.test --nbval tutorial/QuantileRegression/QuantilRegressionForest.ipynb