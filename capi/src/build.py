# This file is part of PDFFlow
import cffi
ffibuilder = cffi.FFI()

with open('pdfflow/pdfflow.h') as f:
    ffibuilder.embedding_api(f.read())

ffibuilder.set_source('cpdfflow', r'''
    #include "pdfflow/pdfflow.h"
''', source_extension='.c')

with open('wrapper.py') as f:
    ffibuilder.embedding_init_code(f.read())

ffibuilder.emit_c_code('cpdfflow.cc')
