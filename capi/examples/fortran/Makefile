# Makefile example

CFLAGS=`pkg-config pdfflow --cflags`
LIBS=`pkg-config pdfflow --libs` 
FC=gfortran

all: example

%: %.f90
	$(FC) $(CFLAGS) -o $@ pdfflow_f_interface.c $< $(LIBS)

run: example
	./example

clean:
	rm -rf example
