for note in $(find . -iname "*ipynb" | grep -v checkpoints); do
    echo jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace ${note}
   jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace ${note}
done

py.test --nbval  $(find . -iname "*ipynb" |  grep -v aux | grep -v TTreeHn | grep -v checkpoints) | tee notebookTest.log
