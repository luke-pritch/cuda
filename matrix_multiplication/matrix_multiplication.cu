#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication
__global__ void matMulKernel(float *A, float *B, float *C, int M, int N, int P) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < P) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * P + col];
        }
        C[row * P + col] = sum;
    }
}

// Host function to perform matrix multiplication
void matMul(float *A, float *B, float *C, int M, int N, int P) {
    // Allocate device memory for matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, M * N * sizeof(float));
    cudaMalloc((void **)&d_B, N * P * sizeof(float));
    cudaMalloc((void **)&d_C, M * P * sizeof(float));

    // Copy host matrices to device
    cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * P * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel with 16x16 thread blocks
    int blockSize = 16;
    int gridDimX = (M + blockSize - 1) / blockSize;
    int gridDimY = (P + blockSize - 1) / blockSize;
    dim3 blockSize3D(blockSize, blockSize, 1);
    dim3 gridDim3D(gridDimX, gridDimY, 1);
    matMulKernel<<<gridDim3D, blockSize3D>>>(d_A, d_B, d_C, M, N, P);

    // Copy result from device to host
    cudaMemcpy(C, d_C, M * P * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int M = 3, N = 4, P = 2;

    // Allocate host matrices
    float *A = (float *)malloc(M * N * sizeof(float));
    float *B = (float *)malloc(N * P * sizeof(float));
    float *C = (float *)malloc(M * P * sizeof(float));

    // Initialize matrices
    for (int i = 0; i < M * N; i++) {
        A[i] = i;
    }
    for (int i = 0; i < N * P; i++) {
        B[i] = i;
    }

    // Perform matrix multiplication
    matMul(A, B, C, M, N, P);

    // Print result
    for (int i = 0; i < M * P; i++) {
        printf("%f ", C[i]);
        if ((i + 1) % P == 0) printf("\n");
    }

    free(A);
    free(B);
    free(C);

    return 0;
}
