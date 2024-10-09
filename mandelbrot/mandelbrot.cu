#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Define the size of the image
#define WIDTH 800
#define HEIGHT 800

// Define the maximum number of iterations
#define MAX_ITERATIONS 256

// Define the size of the workgroup
#define WORKGROUP_SIZE 16

// CUDA kernel for generating the Mandelbrot set
__global__ void mandelbrotKernel(float *image, int width, int height, float xmin, float xmax, float ymin, float ymax) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float c_real = xmin + x * (xmax - xmin) / width;
        float c_imag = ymin + y * (ymax - ymin) / height;

        float z_real = 0.0f;
        float z_imag = 0.0f;
        int iteration = 0;

        while (iteration < MAX_ITERATIONS && (z_real * z_real + z_imag * z_imag) < 4.0f) {
            float temp = z_real * z_real - z_imag * z_imag + c_real;
            z_imag = 2.0f * z_real * z_imag + c_imag;
            z_real = temp;
            iteration++;
        }

        if (iteration == MAX_ITERATIONS) {
            image[y * width + x] = 0.0f;
        } else {
            image[y * width + x] = (float)iteration / MAX_ITERATIONS;
        }
    }
}

int main() {
    // Allocate host memory for the image
    float *image = (float *)malloc(WIDTH * HEIGHT * sizeof(float));

    // Define the boundaries of the Mandelbrot set
    float xmin = -2.5f;
    float xmax = 1.5f;
    float ymin = -1.5f;
    float ymax = 1.5f;

    // Allocate device memory for the image
    float *d_image;
    cudaMalloc((void **)&d_image, WIDTH * HEIGHT * sizeof(float));

    // Copy host image to device
    cudaMemcpy(d_image, image, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel with 2D workgroups
    dim3 blockSize(WORKGROUP_SIZE, WORKGROUP_SIZE, 1);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y, 1);
    mandelbrotKernel<<<gridSize, blockSize>>>(d_image, WIDTH, HEIGHT, xmin, xmax, ymin, ymax);

    // Copy result from device to host
    cudaMemcpy(image, d_image, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);

    // Save the image to a file
    FILE *fp = fopen("mandelbrot.ppm", "wb");
    fprintf(fp, "P6\n%d %d\n255\n", WIDTH, HEIGHT);
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            unsigned char pixel = (unsigned char)(image[y * WIDTH + x] * 255.0f);
            fwrite(&pixel, 1, 1, fp);
            fwrite(&pixel, 1, 1, fp);
            fwrite(&pixel, 1, 1, fp);
        }
    }
    fclose(fp);

    // Free device memory
    cudaFree(d_image);

    // Free host memory
    free(image);

    return 0;
}
