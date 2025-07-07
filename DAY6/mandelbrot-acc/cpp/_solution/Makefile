CXX=nvc++
CXXFLAGS=-fast -acc=gpu -gpu=pinned -Minfo=all -mp

mandelbrot_solution.x: mandelbrot.o main_solution.o
	$(CXX) $(CXXFLAGS) -o $@ $^

.PHONY: clean
clean:
	rm -f *.o core *.lst *.ptx *.pgprof *.cubin *.s *.x *.mod *.nvprof
