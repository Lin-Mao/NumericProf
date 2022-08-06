mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))
all:
	nvcc -ccbin=$(CXX) -O3 -lineinfo -arch=sm_86 iterative_mul_saltation.cu -o $(current_dir).exe

clean:
	rm -f *.exe
