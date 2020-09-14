.. _overview-label:

========
Overview
========

.. contents::
   :local:
   :depth: 1

Installing a PDF set
====================
PDF sets can be installed in two ways

	1. downloading directly the PDF sets, from instance from `LHAPDF PDF sets page <https://lhapdf.hepforge.org/pdfsets.html>`_ or `NNPDF <http://nnpdf.mi.infn.it/for-users/unpolarized-pdf-sets/>`_
	2. exploiting the ``lhapdf`` scripts, through the following commands:

.. code-block:: bash

	lhapdf list
	lhapdf install <pdf set>

which downloads and install the sets directly to ``lhapdf-config --datadir``.


Instantiating a PDF
===================

mkPDF wrapper
^^^^^^^^^^^^^

PDFs can be instantiated in a similar manner to `LHAPDF <https://lhapdf.hepforge.org/>`_
by calling the provided ``mkPDF`` and ``mkPDFs`` functions.

If ``LHAPDF`` and the ``pdfset`` are installed in the system it is enough to call:

.. code-block:: python

  from pdfflow.pflow import mkPDF
  pdf = mkPDF(f"{pdfset}/0")
  
And ``pdfflow`` will try to obtain the PDF directory
from ``LHAPDF``. If, instead, we have manually downloaded the PDF, we need to specify the folder
in which the PDF folder can be found, for instance:

.. code-block:: python

  from pdfflow.pflow import mkPDF
  pdf = mkPDF(f"{pdfset}/0", dirname="/usr/share/lhapdf/LHAPDF")

To obtain the central member (0) of the ``pdfset``.
It is often necessary to require several members of a set, for instance to compute
pdf error. This can be achieved with the ``mkPDFs`` function, for instance,
to obtain members (0,1,2) we can do:

.. code-block:: python

  from pdfflow.pflow import mkPDFs
  pdf = mkPDFs(pdfset, [0, 1, 2])

Note that both ``mkPDF`` and ``mkPDFs`` accept the keyword argument ``dirname``.


PDF class
^^^^^^^^^

The aforementioned functions are all wrappers around the low-level ``PDF`` class and provide an instance to the class.
The class can also be instantiated directly with:

.. code-block:: python

  from pdfflow.pflow import PDF
  pdf = PDF(dirname, pdfset, [0]) # obtain a PDF instance for member 0
  pdf = PDF(dirname, pdfset, [2, 5]) # obtain a PDF instance for members 2 and 5

Note that in order to instantiate a PDF class it is always necessary to provide the source directory of the PDF sets.

PDF UIs usage
=============
The PDF interpolation can be worked out calling the ``py_xfxQ2`` method with
python or TensorFlow objects as arguments:

Python interface
^^^^^^^^^^^^^^^^

When using python arguments as the input we provide the ``py_xfxQ2``.
This function deals with the conversion of the input into TensorFlow variables.

.. code-block:: python

	from pdfflow.pflow import mkPDFs
	
	pdf = mkPDFs(pdfset, [0,1,2])
	x = [10**i for i in range(-6,-1)]
	q2 = [10**(2*i) for i in range(1,6)]
	pid = [-1,21,1]

	pdf.py_xfxQ2(pid, x, q2)
	

TensorFlow interface
^^^^^^^^^^^^^^^^^^^^

Instead, if the arguments are already tensorflow objects, it is possible to call
lower level ``tf.functions`` such as ``xfxQ2``:

.. code-block:: python

	from pdfflow.pflow import mkPDFs
	from pdfflow.configflow import float_me, int_me
	
	pdf = mkPDFs(pdfset, [0,1,2])
	x = float_me([10**i for i in range(-6,-1)])
	q2 = float_me([10**(2*i) for i in range(1,6)])
	pid = int_me([-1,21,1])

	pdf.xfxQ2(pid, x, q2)
	
.. note:: The ``float_me`` and ``int_me`` functions are wrappers around ``tf.cast`` which we provide with the aim of ensuring that integers are cast to 32-bit integers and float to 64-bit floats.

If arguments had been ``tf.Tensor`` objects, the preferred way to call the interpolation would have been
via the ``xfxQ2`` function.
To go through the computation of all the pids in the flavor scheme, use ``xfxQ2_allpid`` or the
``py_xfxQ2_allpid`` version instead.


Strong coupling interpolation
-----------------------------

The strong coupling interpolation requires calling its own methods of the ``PDF`` object:

.. code-block:: python

	from pdfflow.pflow import mkPDFs
	
	pdf = mkPDFs(pdfset, [0,1,2])
	pdf.alphas_trace()

	q2 = [10**(2*i) for i in range(1,6)]
	pdf.py_alphasQ2(q2)

According to the PDF interpolation discussed above, we provide the user with ``py_alphasQ2`` Python and ``alphasQ2`` ``TensorFlow`` interfaces for the strong coupling interpolation.

In order to mimic the ``LHAPDF`` set of functions, we implement also the ``alphasQ`` and ``py_alphasQ`` ``PDF`` methods, by which the user is relieved of squaring the query array elements manually.