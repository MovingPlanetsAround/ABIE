/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include "helper_cuda.h"
#include <math.h>

#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;
// #include "bodysystem.h"

template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

template<typename T>
__device__ T rsqrt_T(T x)
{
    return rsqrt(x);
}

template<>
__device__ float rsqrt_T<float>(float x)
{
    return rsqrtf(x);
}

template<>
__device__ double rsqrt_T<double>(double x)
{
    return rsqrt(x);
}


// Macros to simplify shared memory addressing
#define SX(i) sharedPos[i+blockDim.x*threadIdx.y]
// This macro is only used when multithreadBodies is true (below)
#define SX_SUM(i,j) sharedPos[i+blockDim.x*j]



__device__ double3 bodyBodyInteraction(double3 ai,
                    double4 bi,
                    double4 bj,
                    double G)
{
    double3 r;

    // r_ij  [3 FLOPS]
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
    double distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    if (distSqr == 0) return ai;

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    double invDist = rsqrt(distSqr);  // double point precision
    // double invDist = (double)rsqrtf(distSqr); // float point precision
    double invDistCube =  invDist * invDist * invDist;

    // s = m_j * invDistCube [1 FLOP]
    double s = G * bj.w * invDistCube;

    // a_i =  a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;
}

__device__ double3 computeBodyAccel(double4 bodyPos,
                 double4 *positions,
                 int numTiles, cg::thread_block cta, double G)
{
    double4 *sharedPos = SharedMemory<double4>();

    double3 acc = {0.0f, 0.0f, 0.0f};

    for (int tile = 0; tile < numTiles; tile++)
    {
        sharedPos[threadIdx.x] = positions[tile * blockDim.x + threadIdx.x];

        cg::sync(cta);

        // This is the "tile_calculation" from the GPUG3 article.
#pragma unroll 128

        for (unsigned int counter = 0; counter < blockDim.x; counter++)
        {
            acc = bodyBodyInteraction(acc, bodyPos, sharedPos[counter], G);
        }

        cg::sync(cta);
    }

    return acc;
}

__global__ void
integrateBodies(double4 *oldPos,
                double3 *acc,
                unsigned int deviceOffset, unsigned int deviceNumBodies,
                double G, int numTiles)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= deviceNumBodies)
    {
        return;
    }

    double4 position = oldPos[deviceOffset + index];

    double3 accel = computeBodyAccel(position, oldPos, numTiles, cta, G);

    // store accelerations
    acc[deviceOffset + index]    = accel;
}

void integrateNbodySystem(double4 *dPos, double3 *acc,
                          double G,
                          unsigned int numBodies,
                          int blockSize)
{
    int numBlocks = (numBodies + blockSize-1) / blockSize;
    int numTiles = (numBodies + blockSize - 1) / blockSize;
    int sharedMemSize = blockSize * 4 * sizeof(double); // 4 floats for pos
    int offset = 0;

    integrateBodies<<< numBlocks, blockSize, sharedMemSize >>>(dPos, acc, offset, numBodies, G, numTiles);


    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution failed");

}

/*
// Explicit specializations needed to generate code
template void integrateNbodySystem<float>(float4 *dPos,
                                          float3 *acc,
                                          float G,
                                          unsigned int numBodies,
                                          int blockSize);

template void integrateNbodySystem<double>(double4 *dPos,
                                           double3 *acc,
                                           double G,
                                           unsigned int numBodies,
                                           int blockSize);
                                           */
