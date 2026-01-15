#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "coloringCC_gpu_2.h"

#define THREADS_PER_BLOCK 256 // Define number of threads per block. Can be tuned for better performance.
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {  // macro for error checking CUDA calls.
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

__global__ void kernel_1(int nrows, int *rowptr, int *index, int *labels, int *d_changed) {
  int warp_id = (blockIdx.x*blockDim.x + threadIdx.x) / 32;
  int thread_id = threadIdx.x % 32;
  if (warp_id<nrows) {
    int my_label = labels[warp_id];
    if (my_label == 0) return;

    int start = rowptr[warp_id];
    int end = rowptr[warp_id+1];
    int min_found = my_label;
    for (int i=start+thread_id; i<end; i+=32) {
      int u = index[i];
      int lu = labels[u];
      if (lu < min_found) {
        min_found = lu;
      }
    }

    for (int offset = 16; offset>0; offset /=2) {
      int other_lu = __shfl_down_sync(0xFFFFFFFF, min_found, offset);
      if (other_lu < min_found) min_found = other_lu;
    }

    if (thread_id == 0) {
      if (min_found < my_label) {
        atomicExch(&labels[warp_id], min_found);
        if (*d_changed == 0) *d_changed = 1;
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

void coloringCC_gpu_2(int nrows, int nnz, const int *rowptr, const int *index, int *labels) {
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

  int blocksPerGrid_k1 = (nrows + (THREADS_PER_BLOCK / 32) - 1) / (THREADS_PER_BLOCK / 32);
  int blocksPerGrid_k2 = (nrows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  h_changed = 1;
  while (h_changed !=0) { // The main loop that the algorithm takes place. It continues until no changes happen in an iteration.
    counter_of_while_iterations++;
    h_changed = 0;
    gpuErrchk(cudaMemset(d_changed, 0, sizeof(int)));

    kernel_1<<<blocksPerGrid_k1, THREADS_PER_BLOCK>>>(nrows, d_rowptr, d_index, d_labels, d_changed);
    gpuErrchk(cudaGetLastError());

    kernel_2<<<blocksPerGrid_k2, THREADS_PER_BLOCK>>>(nrows, d_labels, d_changed);
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
  }
  printf("Number of while iteration: %d\n", counter_of_while_iterations); 
  gpuErrchk(cudaMemcpy(labels, d_labels, sizeof(int)*nrows, cudaMemcpyDeviceToHost)); // Copy back the labels vector from device to host.

  cudaFree(d_rowptr), cudaFree(d_index), cudaFree(d_labels), cudaFree(d_changed);
}


