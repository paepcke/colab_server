from setuptools import setup, find_packages
import os
import glob

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "colab_server",
    version = "0.1",
    packages = find_packages(),

    # Dependencies on other packages:
    setup_requires   = [],
    install_requires = ['jupyter_http_over_ws>=0.0.8',
                        'ipython>=6.1.0',
                        'ipywidgets>=7.5.1',
                        'torch>=1.5.0',
                        'keras>=2.3.1',
                        'tensorflow>=2.2.0',
                        'tqdm>=4.46.0',
                        'protobuf>=3.12.2',
                        #'google-colab>=1.0.0', 
                        ],

    #dependency_links = ['https://github.com/DmitryUlyanov/Multicore-TSNE/tarball/master#egg=package-1.0']
    # Unit tests; they are initiated via 'python setup.py test'
    test_suite       = 'nose.collector',
    #test_suite       = 'tests',
    tests_require    =['nose'],

    # metadata for upload to PyPI
    author = "Alyssa Romanos",
    author_email = "paepcke@cs.stanford.edu",
    description = "BERT-analyze Facebook ads",
    long_description_content_type = "text/markdown",
    long_description = long_description,
    license = "BSD",
    keywords = "text analysis",
)


