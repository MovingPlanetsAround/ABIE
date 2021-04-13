# Using `long double` instead of `double` for improved precision
LONGDOUBLE = 0

# Use CUDA/GPU to perform the calculation of acceleratoins
GPU = 1

# CUDA installation path
CUDA_SDK_PATH=/usr/local/cuda


ifeq ($(GPU), 1)
	OBJS = common.o integrator_runge_kutta.o integrator_gauss_radau15.o integrator_wisdom_holman.o additional_forces.o gpuforce.o bodysystemcuda.o
	CFLAGS += -DGPU
else
	OBJS = common.o integrator_runge_kutta.o integrator_gauss_radau15.o integrator_wisdom_holman.o additional_forces.o
endif

DEPS = common.h

ifeq ($(LONGDOUBLE), 1)
	CFLAGS += -DLONGDOUBLE -fPIC -O3 -std=c99 -g
else
	CFLAGS += -fPIC -O3 -std=c99 -g
endif

ifeq ($(GPU), 1)
	NVCC = nvcc
else
	NVCC = $(CC)
endif

libabie.so: $(OBJS)
ifeq ($(GPU), 1)
	$(CC) -o $@ $(OBJS) $(CFLAGS) -shared -L${CUDA_SDK_PATH}/lib64 -I${CUDA_SDK_PATH}/include -lcudart -lstdc++
	# rm *.o
else
	$(CC) -o $@ $(OBJS) $(CFLAGS) -shared 
	rm *.o
endif

gpuforce.o: gpuforce.cu
ifeq ($(GPU), 1)
	$(NVCC) -Xcompiler -fPIC -DGPU -O3 -g $^ -c -o $@
endif

bodysystemcuda.o: bodysystemcuda.cu
ifeq ($(GPU), 1)
	$(NVCC) -Xcompiler -fPIC -DGPU -O3 -g $^ -c -o $@
endif

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

clean:
	rm *.o *.so 
