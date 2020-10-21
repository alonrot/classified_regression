import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "classireg",
    version = "1.0",
    author = "",
    author_email = "",
    description = (""),
    keywords = "Bayesian Optimization, Gaussian process, Active Learning, Safe Learning",
    packages=['classireg',
              'classireg.models',
              'classireg.varinf',
              'classireg.utils',
              'classireg.tests',],
    long_description=read('README.md'),
)