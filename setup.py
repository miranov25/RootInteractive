from distutils.core import setup
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    author_email='marian.ivanov@cern.ch',
    author='Marian Ivanov',
    url='https://github.com/miranov25/RootInteractive',
    name='RootInteractive',
    version='v0.01.00.dev0',
    #packages=setuptools.find_packages(),
    packages=setuptools.find_packages(exclude=["scripts*", "tests*","*d.ts"]),
    license='Not defined yet. Most probably similar to ALICE (CERN)  license',
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=False,
    package_data={
    '': ['../*/*/*/*.ts']
    },
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
        'bokeh>2,<=2.3',
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
        'tensorflow>=2.4.0',
        'keras',
        # ------------------      test and tutorials
        'pytest',
        'nbval',
        'nb-clean'
    ]
)
