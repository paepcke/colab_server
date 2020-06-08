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
                        #'ipython>=6.1.0',
                        'ipython>=5.5.0',
                        'Cython',
                        #'ipywidgets>=7.5.1',
                        #'ipywidgets',
                        'torch>=1.5.0',
                        'keras>=2.3.1',
                        #'tensorflow>=2.2.0',
                        'tensorflow',
                        'tqdm>=4.46.0',
                        'protobuf>=3.12.2',
                        'scikit-learn>=0.23.1',
                        'pytorch-pretrained-bert>=0.6.2',
                        #'pandas>=1.0.3',
                        'pandas>=0.24.2',
                        'matplotlib>=3.2.1',
                        'portpicker>=1.3.1',
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


