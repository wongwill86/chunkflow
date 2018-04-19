========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |requires|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|

.. |docs| image:: https://readthedocs.org/projects/chunkflow/badge/?style=flat
    :target: https://readthedocs.org/projects/chunkflow
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/wongwill86/chunkflow.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/wongwill86/chunkflow

.. |requires| image:: https://requires.io/github/wongwill86/chunkflow/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/wongwill86/chunkflow/requirements/?branch=master

.. |codecov| image:: https://codecov.io/github/wongwill86/chunkflow/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/wongwill86/chunkflow

.. |version| .. image:: https://img.shields.io/pypi/v/chunkflow.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/chunkflow

.. |commits-since| .. image:: https://img.shields.io/github/commits-since/wongwill86/chunkflow/v0.1.0.svg
    :alt: Commits since latest release
    :target: https://github.com/wongwill86/chunkflow/compare/v0.1.0...master

.. |wheel| .. image:: https://img.shields.io/pypi/wheel/chunkflow.svg
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/chunkflow

.. |supported-versions| .. image:: https://img.shields.io/pypi/pyversions/chunkflow.svg
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/chunkflow

.. |supported-implementations| .. image:: https://img.shields.io/pypi/implementation/chunkflow.svg
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/chunkflow


.. end-badges

Convnet Inference

* Free software: Apache Software License 2.0

Installation
============

::

    pip install chunkflow

Documentation
=============

https://chunkflow.readthedocs.io/

Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
