[![Documentation Status](https://readthedocs.org/projects/pdfflow/badge/?version=latest)](https://pdfflow.readthedocs.io/en/latest/?badge=latest)
![pytest](https://github.com/N3PDF/pdfflow/workflows/pytest/badge.svg)
[![DOI](https://zenodo.org/badge/238731330.svg)](https://zenodo.org/badge/latestdoi/238731330)
[![AUR](https://img.shields.io/aur/version/python-pdfflow)](https://aur.archlinux.org/packages/python-pdfflow)


# PDFflow

PDFflow is parton distribution function interpolation library written in Python and based on the [TensorFlow](https://www.tensorflow.org/) framework. It is developed with a focus on speed and efficiency, enabling researchers to perform very expensive calculation as quick and easy as possible.

The key features of PDFflow is the possibility to query PDF sets on GPU accelerators.

## Documentation

[https://pdfflow.readthedocs.io/en/latest](https://pdfflow.readthedocs.io/en/latest)

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

## Examples

There are some examples in the `benchmarks` folder.

## Citation policy

If you use the package pelase cite the following paper and zenodo references:
- https://doi.org/10.5281/zenodo.3964190
- https://arxiv.org/abs/20xx.xxxxx
