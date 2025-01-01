all: km_pthreads km_openmp

km_pthreads: km_pthreads.o
	g++ -o km_pthreads km_pthreads.o -pthread

km_pthreads.o: km_pthreads.cpp
	g++ -c -o km_pthreads.o km_pthreads.cpp -pthread

km_openmp: km_openmp.o
	g++ -o km_openmp km_openmp.o -fopenmp

km_openmp.o: km_openmp.cpp
	g++ -c -o km_openmp.o km_openmp.cpp -fopenmp

clean:
	rm  km_pthreads.o km_openmp.o
