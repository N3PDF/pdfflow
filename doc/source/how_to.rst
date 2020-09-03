.. _howto-label:

==========
How to use
==========

Building the graph ahead of time
================================

The very first iteration of PDFFlow compiles the ``tf.Graph``. TensorFlow compiles only functions that are called for the first time. The function ``PDF.trace()`` is intended to build all the necessary parts of the ``tf.Graph`` and prevent future retracings that could slow down the execution.

Then a ``PDF`` object can be instantiated by the following lines of code:

``p = pdf.mkPDF(pdfname, DIRNAME)``
``p.trace()``


The strong coupling interpolation requires calling the equivalent ``PDF.alphas_trace()`` function instead:

``p = pdf.mkPDF(pdfname, DIRNAME)``
``p.alphas_trace()``