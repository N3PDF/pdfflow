.. _overview-label:

========
Overview
========

.. contents::
   :local:
   :depth: 1

Instantiating a PDF
-------------------
PDFs can be instantiated in a similar manner to [``LHAPDF``](https://lhapdf.hepforge.org/)
by calling the provided ``mkPDF`` and ``mkPDFs`` functions.

If ``LHAPDF`` and the ``pdfset`` are installed in the system it is enough to call:

.. code-block:: python

  from pdfflow.pflow import mkPDF
  pdf = mkPDF(f"{pdfset}/0")


To obtain the central member (0) of the ``pdfset``.
It is often necessary to require several members of a set, for instance to compute
pdf error. This can be achieved with the ``mkPDFs`` function, for instance,
to obtain members (0,1,2) we can do:

.. code-block:: python

  from pdfflow.pflow import mkPDF
  pdf = mkPDFs(pdfset, [0, 1, 2])


Note that both ``mkPDF`` and ``mkPDFs`` accept the keyword argument ``dirname``.
If ``dirname`` is not provided, ``pdfflow`` will try to obtain the PDF directory
from ``LHAPDF``.
If ``dirname`` is provided, instead, ``pdfflow`` will load the pdffset from the given directory.
These functions are all wrappers around the low-level ``PDF`` class and provide an instance to the class.
The class can also be instantiated directly with:

.. code-block:: python

  from pdfflow.pflow import PDF
  pdf = PDF(dirname, pdfset, [0]) # obtain a PDF instance for member 0
  pdf = PDF(dirname, pdfset, [2, 5]) # obtain a PDF instance for members 2 and 5
