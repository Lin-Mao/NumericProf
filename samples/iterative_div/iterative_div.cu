#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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
__global__ void iterative_mul(float *d_c, float *d_a, int n) {
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n) {
        for (int i=0; i < 20; i++) {
            d_c[id] /= d_a[id];
        }
    }
}

int main(int argc, char *argv[]) {
    // Size of vectors
    int n = 100;
    if (argc > 1) n = atoi(argv[1]);

    // Host vectors
    float *h_a;
    float *h_c;

    // Device vectors
    float *d_a;
    float *d_c;

    // Size, in bytes, of each vector
    size_t bytes = n * sizeof(float);

    // Allocate memory for each vector on host
    h_a = (float *)malloc(bytes);
    h_c = (float *)malloc(bytes);

    // Allocate memory for each vector on GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_c, bytes);

    int i;
    // Initialize vectors on host
    for (i = 0; i < n; i++) {
        h_a[i] = sin(i);
        h_c[i] = sin(i);
    }

    // Copy host vectors to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, bytes, cudaMemcpyHostToDevice);

    int blockSize, gridSize;

    // Number of threads in each thread block
    blockSize = 64;
    // Number of thread blocks in grid
    gridSize = (int)ceil((float)n / blockSize);

    // Execute the kernel
    CUDA_SAFECALL((iterative_mul<<<gridSize, blockSize>>>(d_c, d_a, n)));

    cudaDeviceSynchronize();

    // Copy array back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Sum up vector c and print result divided by n, this should equal 1 within
    // error
    float sum = 0;
    for (i = 0; i < n; i++) printf("%f\n", h_c[i]);
    printf("Final sum = %f; sum/n = %f (should be ~1)\n", sum, sum / n);

    // Release device memory
    cudaFree(d_a);
    cudaFree(d_c);

    // Release host memory
    free(h_a);
    free(h_c);

    return 0;
}
