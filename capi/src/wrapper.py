# This file is part of
from cpdfflow import ffi


@ffi.def_extern()
def hello():
    print('Hello!!!')