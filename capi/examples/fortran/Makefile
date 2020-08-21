# Makefile example

CFLAGS=`pkg-config pdfflow --cflags` -fno-underscoring
LIBS=`pkg-config pdfflow --libs` 
FC=gfortran

all: example

%: %.f90
	$(FC) $(CFLAGS) -o $@ $< $(LIBS)

run: example
	./example

clean:
	rm -rf example
