#clean_ipynb */*/*ipynb
for note in $(ls */*/*ipynb); do
    echo clean  ${note}
    nb-clean clean < ${note}  > ${note}2
    mv ${note}2 ${note}
done

py.test --nbval  $(ls */*/*ipynb | grep -v aux | grep -v TTreeHn) | tee pytest.log