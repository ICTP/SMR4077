FC=nvfortran
FCFLAGS=-fast -acc=gpu -gpu=pinned -Minfo=all -mp

mandelbrot_solution.x: mandelbrot.o main_solution.o
	$(FC) $(FCFLAGS) -o $@ $^

.SUFFIXES: .o .F90
.F90.o: 
	$(FC) $(FCFLAGS) -c -o $@ $<

.PHONY: clean
clean:
	rm -f *.o core *.lst *.ptx *.pgprof *.cubin *.s *.x *.mod *.nvprof
