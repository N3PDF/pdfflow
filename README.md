[![DOI](https://zenodo.org/badge/238731330.svg)](https://zenodo.org/badge/latestdoi/238731330)
[![arxiv](https://img.shields.io/badge/arXiv-hep--ph%2F2009.06635-%23B31B1B.svg)](https://arxiv.org/abs/2009.06635)

[![Documentation Status](https://readthedocs.org/projects/pdfflow/badge/?version=latest)](https://pdfflow.readthedocs.io/en/latest/?badge=latest)
![pytest](https://github.com/N3PDF/pdfflow/workflows/pytest/badge.svg)
[![AUR](https://img.shields.io/aur/version/python-pdfflow)](https://aur.archlinux.org/packages/python-pdfflow)


# PDFFlow

PDFFlow is parton distribution function interpolation library written in Python and based on the [TensorFlow](https://www.tensorflow.org/) framework. It is developed with a focus on speed and efficiency, enabling researchers to perform very expensive calculation as quick and easy as possible.

The key features of PDFFlow is the possibility to query PDF sets on GPU accelerators.

## Documentation

The documentation for PDFFlow can be consulted in the readthedocs page: [pdfflow.readthedocs.io](https://pdfflow.readthedocs.io/en/latest).

## Installation

The package can be installed with pip:
```
python3 -m pip install pdfflow
```

If you prefer a manual installation just use:
```
python setup.py install
```
or if you are planning to extend or develop code just use:
```
python setup.py develop
```

## Minimal Working Example

Below a minimalistic example where `PDFFlow` is used to generate a 10 values of the PDF
for 2 members for three different flavours.

```python
from pdfflow import mkPDFs
import tensorflow as tf

pdf = mkPDFs("NNPDF31_nnlo_as_0118", [0,2])
x = tf.random.uniform([10], dtype=tf.float64)
q2 = tf.random.uniform([10], dtype=tf.float64)*20 + 10
pid = tf.cast([-1,21,1], dtype=tf.int32)

result = pdf.xfxQ2(pid, x, q2)
```

Note the usage of the `dtype` keyword inm the TensorFlow calls.
This is used to ensure that `float64` is being used all across the program.
For convenience, we ship two functions, `int_me` and `float_me` which are simply
wrappers to `tf.cast` with the right types.

These wrappers can be used over TensorFlow types but also numpy values:
```python
from pdfflow import mkPDFs, int_me, float_me
import tensorflow as tf
import numpy as np

pdf = mkPDFs("NNPDF31_nnlo_as_0118", [0,2])
x = float_me(np.random.rand(10))
q2 = float_me(tf.random.uniform([10])*20 + 10)
pid = int_me([-1,21,1])

result = pdf.xfxQ2(pid, x, q2)
```

## Citation policy

If you use the package pelase cite the following paper and zenodo references:
- https://doi.org/10.5281/zenodo.3964190
- https://arxiv.org/abs/2009.06635
