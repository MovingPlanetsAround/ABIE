#ifdef GPU

extern "C" {

#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include "common.h"

}

#include "helper_cuda.h"


#define BLOCK_SIZE 256
#define SOFTENING 0.0f

double3 *pos_dev;
double3 *acc_dev;
double *masses_dev;

int inited = 0;



void calculate_force_cuda(double4* oldPos, double G, int numBodies, double4* acc);
__global__ void cudaforce(double4* oldPos, double G, int numBodies, double3* acc);
void integrateNbodySystem(double4 *dPos, double3 *acc,
                           double G,
                           unsigned int numBodies,
                           int blockSize);

template<typename T>
__global__ void gpuforce(double4 *p, T G, int n, double3 *acc) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        T Fx = 0.0f; T Fy = 0.0f; T Fz = 0.0f;

#pragma unroll
        for (int j = 0; j < n; j++) {
            T m = p[j].w;
            if (i == j || m == 0) continue;
            T dx = p[i].x - p[j].x;
            T dy = p[i].y - p[j].y;
            T dz = p[i].z - p[j].z;
            T distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            if (distSqr == SOFTENING) continue;
            T invDist = rsqrt(distSqr);
            T invDist3 = invDist * invDist * invDist;

            Fx -= (G * m * dx * invDist3);
            Fy -= (G * m * dy * invDist3);
            Fz -= (G * m * dz * invDist3);
        }
        acc[i].x = Fx; acc[i].y = Fy; acc[i].z = Fz;
    }
}

template<typename T>
__global__ void gpuforce_v2(double3 *p, T *masses, T G, int n, double3 *acc) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        T Fx = 0.0f; T Fy = 0.0f; T Fz = 0.0f;

#pragma unroll
        for (int j = 0; j < n; j++) {
            T m = masses[j];
            if (i == j || m == 0) continue;
            T dx = p[i].x - p[j].x;
            T dy = p[i].y - p[j].y;
            T dz = p[i].z - p[j].z;
            T distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            if (distSqr == SOFTENING) continue;
            T invDist = rsqrt(distSqr);
            T invDist3 = invDist * invDist * invDist;

            Fx -= (G * m * dx * invDist3);
            Fy -= (G * m * dy * invDist3);
            Fz -= (G * m * dz * invDist3);
        }
        acc[i].x = Fx; acc[i].y = Fy; acc[i].z = Fz;
    }
}

extern "C" {

    void gpu_init(int N, int deviceID) {
        printf("Device id, %d\n", deviceID);
        if (inited) return;
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            printf("No CUDA device found. Disable GPU acceleration...\n");
            sim.devID = -1;
        } else if (deviceID < device_count) {
            if (deviceID >= 0) {
                // initialize the GPU device 
                sim.devID = gpuDeviceInit(deviceID);
                printf("Device ID = %d, total number of GPU devices: %d\n", sim.devID, device_count);

                // allocate GPU memory
                int bytes = N * sizeof(double3);
                checkCudaErrors(cudaMalloc(&pos_dev, bytes));
                checkCudaErrors(cudaMalloc(&acc_dev, N * sizeof(double3)));
                checkCudaErrors(cudaMalloc(&masses_dev, N * sizeof(double)));

                inited = 1;
                printf("GPU force opened.\n");
            } else {
                // the user chooses to use CPU only because deviceID < 0
                printf("GPU acceleration disabled by the user (deviceID = %d)", deviceID);
                sim.devID = deviceID;
            }
        } else {
            printf("Invalid CUDA device ID. Number of devices: %d, given device ID: %d. Disable GPU acceleration...\n", device_count, deviceID);
            sim.devID = -1;
        }
    }

    void gpu_finalize() {
        printf("Closing CPU force...");
        if (pos_dev != NULL) cudaFree(pos_dev);
        if (acc_dev != NULL) cudaFree(acc_dev);
        printf("done.\n");
    }


    size_t ode_n_body_second_order_gpu(const real vec[], size_t N, real G, const real masses[], const real radii[], real acc[]) {
        if (masses == NULL) {printf("masses=NULL, exiting...\n"); exit(0);}
        double * pos_host = (double *)malloc(N * 3 * sizeof(double));
        //for (size_t i = 0; i < N; i++) {
        //    pos_host[3 * i] = vec[3 * i];
        //    pos_host[3 * i + 1] = vec[3 * i + 1];
        //    pos_host[3 * i + 2] = vec[3 * i + 2];
        //    // pos_host[4 * i + 3] = masses[i];
        //}

        cudaError_t err;


        checkCudaErrors(cudaMemcpy(pos_dev, vec, N*sizeof(double3), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(masses_dev, masses, N*sizeof(double), cudaMemcpyHostToDevice));

        int actual_block_size = BLOCK_SIZE;
        int nBlocks = (N + actual_block_size - 1) / actual_block_size;

        gpuforce_v2<double><<<nBlocks, actual_block_size>>>(pos_dev, masses_dev, (double) G, (int) N, acc_dev);
        // integrateNbodySystem(pos_dev, acc_dev, G, N, BLOCK_SIZE);
        //cudaforce<<<nBlocks, actual_block_size, shm_size>>>(pos_dev, (double) G, (int) N, acc_dev);

        err = cudaGetLastError();
        if (err != cudaSuccess) {printf("Error: %d %s\n", err, cudaGetErrorString(err)); exit(0);}


        checkCudaErrors(cudaMemcpy(acc, acc_dev, N*sizeof(double3), cudaMemcpyDeviceToHost));
        /*
        for (size_t i = 0; i < 3*N; i++) {
            acc[i] = (real) acc_host[i];
        }*/
        

        // printf("\n");
        // for (int i = 0; i < 3 * N; i++) printf("%f\t", acc[i]);
        // exit(0);
        // for (int i = 0; i < 3 * N; i++) acc[i] = (real) acc_host[i];
        // free(pos_host);

        return 0;
    }
} // end extern C

#endif
