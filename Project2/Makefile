# Makefile for building qs_mpi with mpicxx

all: qs_mpi

qs_mpi: qs_mpi.o
	mpicxx -o qs_mpi qs_mpi.o

qs_mpi.o: qs_mpi.cpp
	mpicxx -c -o qs_mpi.o qs_mpi.cpp

clean:
	rm -f qs_mpi.o qs_mpi
