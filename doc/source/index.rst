.. title::
    pdfflow's documentation!

====================================================
PDFflow: PDF interpolation for hardware accelerators
====================================================

.. image:: https://img.shields.io/badge/arXiv-hep--ph%2F2009.06635-%23B31B1B.svg
   :target: https://arxiv.org/abs/2009.06635

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3964190.svg
   :target: https://doi.org/10.5281/zenodo.3964190

.. contents::
   :local:
   :depth: 1

PDFflow is a parton distribution function interpolation library written in Python and based on the `TensorFlow <https://www.tensorflow.org/>`_ framework.
It is developed with a focus on speed and efficiency, enabling researchers to perform very expensive calculation as quick and easy as possible.


How to obtain the code
======================

Open Source
-----------
The ``pdfflow`` package is open source and available at https://github.com/N3PDF/pdfflow

Installation
------------
The package can be installed with pip:

.. code-block:: bash

  python3 -m pip install pdfflow

If you prefer a manual installation just use:

.. code-block:: bash

  git clone https://github.com/N3PDF/pdfflow
  cd pdfflow
  python3 setup.py install

or if you are planning to extend or develop code just use:

.. code-block:: bash

  python3 setup.py develop


Motivation
==========

PDFflow is developed within the `Particle Physics group <http://tiflab.mi.infn.it/>`_ of the `University of Milan <https://www.unimi.it>`_.
Theoretical calculations in particle physics are incredibly time consuming operations, sometimes taking months in big clusters all around the world.

These expensive calculations are driven by the high dimensional phase space that need to be integrated but also by a lack of expertise in new techniques on high performance computation.
Indeed, while at the theoretical level these are some of the most complicated calculations performed by mankind; at the technical level most of these calculations are performed using very dated code and methodologies that are unable to make us of the available resources.

With PDFflow we aim to fill this gap between theoretical calculations and technical performance by providing a framework which can automatically make the best of the machine in which it runs.
To that end PDFflow is based on two technologies that together will enable a new age of research.



How to cite ``pdfflow``?
=========================

When using ``pdfflow`` in your research, please cite the following publications:



.. image:: https://img.shields.io/badge/arXiv-hep--ph%2F2009.06635-%23B31B1B.svg
   :target: https://arxiv.org/abs/2009.06635

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3691926.svg
   :target: https://doi.org/10.5281/zenodo.3691926

Bibtex:

.. code-block:: latex

    @article{Carrazza:2020qwu,
        author = "Carrazza, Stefano and Cruz-Martinez, Juan M. and Rossi, Marco",
        title = "{PDFFlow: parton distribution functions on GPU}",
        eprint = "2009.06635",
        archivePrefix = "arXiv",
        primaryClass = "hep-ph",
        month = "9",
        year = "2020"
    }

    @software{pdfflow_package,
        author       = {Juan Cruz-Martinez and
                        Marco Rossi and
                        Stefano Carrazza},
        title        = {N3PDF/pdfflow: PDFFlow 1.0},
        month        = sep,
        year         = 2020,
        publisher    = {Zenodo},
        version      = {v1.0},
        doi          = {10.5281/zenodo.3964190},
        url          = {https://doi.org/10.5281/zenodo.3964190}
    }	


FAQ
===

Why the name ``pdfflow``?
---------------------------

It is a combination of the names `PDF` and `Tensorflow`.

- **PDFs**: Parton Distribution Functions (or PDFs) are at the core of LHC phenomenology by providing a description of the parton content of the proton.

- **TensorFlow**: the framework developed by Google and made public in November of 2015 is a perfect combination between performance and usability. With a focus on Deep Learning, TensorFlow provides an algebra library able to easily run operations in many different devices: CPUs, GPUs, TPUs with little input by the developer. Write your code once.


.. toctree::
   :maxdepth: 3
   :glob:
   :caption: Contents:

   PDFFlow<self>
   overview
   how_to


.. automodule:: pdfflow
    :members:
    :noindex:
