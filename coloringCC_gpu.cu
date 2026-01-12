#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "coloringCC_gpu.h"

#define THREADS_PER_BLOCK 256 // Define number of threads per block. Can be tuned for better performance.
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {  // macro for error checking CUDA calls.
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

__global__ void kernel_1(int nrows, int *rowptr, int *index, int *labels, int *d_changed) { // First kernel of the algorithn. Each thread processses a vertex.
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < nrows) {
    int my_label = labels[idx];
    int start = rowptr[idx];
    int end = rowptr[idx+1];
    for (int i=start; i<end; i++) {
      int u = index[i];
      int lu = labels[u];
      if (lu < my_label) {
        int old_val = atomicMin(&labels[idx], lu);
        if (lu < old_val) {
          if (*d_changed == 0) { 
            *d_changed = 1;         
          }
          my_label = lu;
        }
      }
    }
  }
}

__global__ void kernel_2(int nrows, int *labels, int *d_changed) {  // Second kernel (pointer jumping) of the algorithm. Imrpoves a lot the convergence speed.
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < nrows) {
    int my_label = labels[idx];
    int new_label = labels[my_label];
    if (new_label < my_label) {
      labels[idx] = new_label;
      if (*d_changed == 0) {
        *d_changed = 1;         
      }
    }
  }
}

void coloringCC_gpu(int nrows, int nnz, const int *rowptr, const int *index, int *labels) {
  int *d_labels = NULL, *d_changed = NULL, *d_rowptr = NULL, *d_index = NULL;
  int h_changed;

  int counter_of_while_iterations = 0;

  gpuErrchk(cudaMalloc((void **)&d_rowptr, sizeof(int)*(nrows+1))); // Allocate device memory
  gpuErrchk(cudaMalloc((void **)&d_index, sizeof(int)*nnz));
  gpuErrchk(cudaMalloc((void **)&d_labels, sizeof(int)*nrows));
  gpuErrchk(cudaMalloc((void **)&d_changed, sizeof(int)));

  gpuErrchk(cudaMemcpy(d_rowptr, rowptr, sizeof(int)*(nrows+1), cudaMemcpyHostToDevice)); // Copy data from host to device
  gpuErrchk(cudaMemcpy(d_index, index, sizeof(int)*nnz, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_labels, labels, sizeof(int)*nrows, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemset(d_changed, 0, sizeof(int)));

  int blocksPerGrid = (nrows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;  // Calculate number of blocks on per grid, depending on the number of nrows and threads per block.
  h_changed = 1;
  while (h_changed !=0) { // The main loop that the algorithm takes place. It continues until no changes happen in an iteration.
    counter_of_while_iterations++;
    h_changed = 0;
    gpuErrchk(cudaMemset(d_changed, 0, sizeof(int)));

    kernel_1<<<blocksPerGrid, THREADS_PER_BLOCK>>>(nrows, d_rowptr, d_index, d_labels, d_changed);
    gpuErrchk(cudaGetLastError());

    kernel_2<<<blocksPerGrid, THREADS_PER_BLOCK>>>(nrows, d_labels, d_changed); // Through experiments, calling the kernel_2 only once per while iteration seems to be the best choice.
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
  }
  printf("Number of while iteration: %d\n", counter_of_while_iterations); 
  gpuErrchk(cudaMemcpy(labels, d_labels, sizeof(int)*nrows, cudaMemcpyDeviceToHost)); // Copy back the labels vector from device to host.

  cudaFree(d_rowptr), cudaFree(d_index), cudaFree(d_labels), cudaFree(d_changed);
}


// In the report i should mention that some graphs like mawi (which might be correlated with the small number of CC) the while loop does only a few (8) while iterations,
// but other gpaphs like road_central need hundreds of iterations (700+). This is probably due to the structure of the graph and the number of CCs. SHOULD DEFINITELY MENTION IT!!