# Compilation sur un noeud GH200 de ROMEO (aarch64), module nvhpc 24.11 :
#   module load nvhpc/nvhpc-nompi/24.11 && make
NVCC    := nvcc
NVCXX   := nvc++
ARCH    := 90
CUFLAGS := -O3 -std=c++17 -arch=sm_$(ARCH)
OMPFLAGS:= -O3 -std=c++17 -mp=gpu -gpu=cc$(ARCH) -Minfo=mp

BIN := bin/cg_cuda bin/cg_omp bin/cg_omp_um

all: $(BIN)

bin/cg_cuda: src/cg_cuda.cu src/common.h
	@mkdir -p bin
	$(NVCC) $(CUFLAGS) -o $@ $< -lcusparse -lcublas

# memoire device explicite (defaut NVHPC sur GH200 : -gpu=mem:separate)
bin/cg_omp: src/cg_omp.cpp src/common.h
	@mkdir -p bin
	$(NVCXX) $(OMPFLAGS) -gpu=mem:separate -o $@ $< 2> bin/minfo_omp.txt || (cat bin/minfo_omp.txt; false)

# memoire unifiee : tout pointeur hote est accessible depuis le GPU
bin/cg_omp_um: src/cg_omp.cpp src/common.h
	@mkdir -p bin
	$(NVCXX) $(OMPFLAGS) -gpu=mem:unified -DOMP_UNIFIED_BUILD -o $@ $< 2> bin/minfo_omp_um.txt || (cat bin/minfo_omp_um.txt; false)

clean:
	rm -rf bin
