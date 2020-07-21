from distutils.core import setup
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    author_email='marian.ivanov@cern.ch',
    author='Marian Ivanov',
    url='https://github.com/miranov25/RootInteractive',
    name='RootInteractive',
    version='v0.00.20b',
    #packages=setuptools.find_packages(),
    packages=setuptools.find_packages(exclude=["scripts*", "tests*"]),
    license='Not defined yet. Most probably similar to ALICE (CERN)  license',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'anytree',
        'nbval',
        'tabulate',
        ## root dependencies
        'uproot',
        'iminuit',
        #'root_pandas',
        ##---------------------   graphics  dependencies
        'bokeh<2,>1',
        'matplotlib',
        'plotly',
        'qgrid',
        'bqplot',
        'beakerx',
        # ----------------------   jupyter notebook dependencies
        'ipywidgets',
        'runtime',
        'requests',
        # ---------------------    machine learning dependencies
        'sklearn',
        # 'scikit-garden',
        'scikit-hep',
        'forestci',
        'tensorflow',
        'keras',
        # ------------------      test and tutorials
        'pytest',
        'nbval',
        'nb-clean'
    ]
)
