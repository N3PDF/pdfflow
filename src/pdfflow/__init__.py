"""PDF interpolation with tensorflow"""
# Expose mkPDF and the int_me, float_me functions
# that way the log system is imported from the very beginning
from importlib.metadata import metadata
from pdfflow.configflow import int_me, float_me, run_eager
from pdfflow.pflow import mkPDF, mkPDFs

PACKAGE = "pdfflow"

__version__ = metadata(PACKAGE)["version"]
