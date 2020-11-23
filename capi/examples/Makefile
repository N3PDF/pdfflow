# Makefile example

CFLAGS=`pkg-config pdfflow --cflags`
LIBS=`pkg-config pdfflow --libs`

all: example

%: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	rm -rf example
