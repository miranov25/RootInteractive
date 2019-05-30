from distutils.core import setup
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    author_email='marian.ivanov@cern.ch',
    author='Marian Ivanov',
    url='https://github.com/miranov25/RootInteractive',
    name='RootInteractive',
    version='v0.00.06',
    #packages=setuptools.find_packages(),
    packages=setuptools.find_packages(exclude=["scripts*", "tests*"]),
    package_data=setuptools.get_package_data(),
    license='Not define yet. Most probably similar to ALICE (CERN)  license',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'anytree',
        'pytest',
        ##---------------------   graphics  dependencies
        'bokeh',
        'matplotlib',
        'plotly',
        'qgrid',
        'bqplot',
        'beakerx',
        # ----------------------   jupyter notebook dependencies
        'ipywidgets',
        'runtime',
        'request',
        # ---------------------    machine learning dependencies
        'sklearn',
        'scikit-garden',
        'forestci',
        'tensorflow',
        'keras'
    ]
)
