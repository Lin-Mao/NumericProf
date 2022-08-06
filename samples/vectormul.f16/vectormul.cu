#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_SAFECALL(call)                                                 \
    {                                                                       \
        call;                                                               \
        cudaError err = cudaGetLastError();                                 \
        if (cudaSuccess != err) {                                           \
            fprintf(                                                        \
                stderr,                                                     \
                "Cuda error in function '%s' file '%s' in line %i : %s.\n", \
                #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            fflush(stderr);                                                 \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }


// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(__half *a, __half *b, __half *c, int n) {
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n) c[id] = __hmul(a[id], b[id]);
}

int main(int argc, char *argv[]) {
    // Size of vectors
    int n = 100000;
    if (argc > 1) n = atoi(argv[1]);

    // Host input vectors
    __half *h_a;
    __half *h_b;
    // Host output vector
    __half *h_c;

    // Device input vectors
    __half *d_a;
    __half *d_b;
    // Device output vector
    __half *d_c;

    // Size, in bytes, of each vector
    size_t bytes = n * sizeof(__half);

    // Allocate memory for each vector on host
    h_a = (__half *)malloc(bytes);
    h_b = (__half *)malloc(bytes);
    h_c = (__half *)malloc(bytes);

    // Allocate memory for each vector on GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    int i;
    // Initialize vectors on host
    for (i = 0; i < n; i++) {
        h_a[i] = __float2half(1);
        h_b[i] = __float2half(1);
        h_c[i] = __float2half(1);
    }

    // Copy host vectors to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, bytes, cudaMemcpyHostToDevice);

    int blockSize, gridSize;

    // Number of threads in each thread block
    blockSize = 1024;

    // Number of thread blocks in grid
    gridSize = (int)ceil((float)n / blockSize);

    // Execute the kernel
    CUDA_SAFECALL((vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n)));

    // Copy array back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Sum up vector c and print result divided by n, this should equal 1 within
    // error
    float sum = 0;
    for (i = 0; i < n; i++) sum += float(h_c[i]);
    printf("Final sum = %f; sum/n = %f (should be ~1)\n", sum, sum / n);

    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
