PDFFlow C-API
=============

This repository contains a C library to access PDFFlow from programming languages different from Python.

## Installation

Make sure you have installed the `pdfflow` module in Python, then in order to install proceed with the usual cmake steps:
```bash
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=<your install prefix>
make
make install
```

## Usage

The compiler flags to include this library in your package can be
retrieved with:
```bash
pkg-config pdfflow --cflags
pkg-config pdfflow --libs
```

Sample programs using this library are provided in the `capi/examples/` directory.
