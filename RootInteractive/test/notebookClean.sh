for note in $(ls */*/*ipynb); do
   echo jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace ${note}
   jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace ${note}
done


