from distutils.core import setup
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    author_email='marian.ivanov@cern.ch, ivanov6@uniba.sk',
    author='Marian I. Ivanov, Marian Ivanov  ',
    url='https://github.com/miranov25/RootInteractive',
    name='RootInteractive',
    version='v0.01.09',
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
        'iminuit',
        ##---------------------   graphics  dependencies
        'bokeh>2,<=2.3',
        'matplotlib',
        # ----------------------   jupyter notebook dependencies
        'ipywidgets',
        'runtime',
        'requests',
        # ---------------------    machine learning dependencies
        'sklearn',
        'scikit-hep',
        "jinja2==3.0.3",
        # ------------------      dependencies needed for test and tutorials in special requirement file requirement_Devel.txt
        "jupyter"                   # not dependence but needed to run tutorials
    ]
)
