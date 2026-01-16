# ColoringCC CUDA Implementation

This project implements the parallel Connected Components labeling algorithm using CUDA for large sparse matrices.

## Dependencies
* **matio** library (installed locally)
* **zlib** & **HDF5** (matio dependencies)

> **DEPENDENCY INSTALLATION NOTE (IMPORTANT):**
  On the Aristotle cluster, the matio library and its dependencies were not available system-wide:
  Therefore, the following libraries were installed locally by me:
    $HOME/local/zlib
    $HOME/local/hdf5
    $HOME/local/matio
The code will not run unless these libraries are available and properly linked

## How to Run
1. **Build:** `make all` 
2. **Setup:** Place your `.mat` files in a `/matrices` directory.
3. **Configure:** Edit `INPUT_FILE_NAME` in `bash_script_gpu_final_test....sh`
4. **Submit:** `sbatch bash_script_gpu_final_test....sh`
5. **Output:** On the .out file
6. **Errors:** on the .err file
> *IMPORTANT:*
  (Supposing that the matio library and dependencies were downloaded on the right foler and in the right way) \
  If you run the program on a HPC cluster, all you need to do is to:
  sbatch bash_script_gpu_final_test_TPN.sh OR sbatch bash_script_gpu_final_test_WPN.sh,
  and everything will happend automatically. There will be a job_(the id of the job).out
  and job_(the id of the job).err. If you dont work on the aristotle HPC cluster, run: make all,
  and there will two executables: executable_TPN, and executable_WPN. You will just need to
  type on the terminal: ./executable_... (the graph that you want).mat, and you are ready to go.

## What i recommend for running the code:
I recommend running the program on the google colab, because it's the easiest way.
You don't need to set the matio libray and its dependencies manually. At the very top cell you just
type the commands: `!apt-get update \ !apt-get install -y libmatio-dev`,
and everything about the matio  library is set. Then you'll need to download the graph using wget... so here there is no need for 
creating the "matrices" directory. On the next cell, at the very start of it you type  
`%%writefile program.cu` and below you copy the code of main_gpu_TPN.cu and coloringCC_gpu_TPN.cu OR main_gpu_WPN.cu and coloringCC_gpu_WPN.cu. 
At the last cell, you just type:`!nvcc -O3 -arch=sm_70 program.cu -lmatio program` and at the end you type:
`!./program (the name of the graph)` and it's done. below there is an example of how the colab version should look like.

```
!apt-get update
!apt-get install -y libmatio-dev

!wget https://suitesparse-collection-website.herokuapp.com/mat/MAWI/mawi_201512020330.mat

%%writefile program.cu
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <matio.h>

#define THREADS_PER_BLOCK 256 // Define number of threads per block. Can be tuned for better performance.
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}
 

__global__ void kernel_1(int nrows, int *rowptr, int *index, int *labels, int *changed) {
  int warp_id = ((long long)blockIdx.x* (long long)blockDim.x + (long long )threadIdx.x) / 32;
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
        if (*changed == 0) *changed = 1;
      }
    }
  }
}

__global__ void kernel_2(int nrows, int *labels, int *changed) {  // Second kernel (pointer jumping) of the algorithm. Imrpoves a lot the convergence speed.
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < nrows) {
    int my_label = labels[idx];
    int new_label = labels[my_label];
    if (new_label < my_label) {
      labels[idx] = new_label;
      if (*changed == 0) {
        *changed = 1;
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


#define memErrchk(ptr) { memAssert((ptr), __FILE__, __LINE__); }
inline void memAssert(void *ptr, const char *file, int line, bool abort=true) { // Macro for error checking memory allocations. Makes the code more readable and easier to debug.
    if (ptr == NULL) {
        fprintf(stderr, "Memory Allocation Failed at: %s:%d\n", file, line);
        if (abort) exit(EXIT_FAILURE);
    }
}

int compare_labels(const void *a, const void *b) {
    int arg1 = *(const int*)a;
    int arg2 = *(const int*)b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

int main(int argc, char* argv[]) {

    struct timeval start;
    struct timeval end;

// ========== Read MAT File and Extract Sparse Matrix ==========
    if (argc < 2) {
        printf("Usage: %s <mat_file>\n", argv[0]);
        return 1;
    }

    mat_t *matfp = Mat_Open(argv[1], MAT_ACC_RDONLY);
    if (matfp == NULL) {
        printf("Cannot open MAT file %s\n", argv[1]);
        return 1;
    }

    matvar_t *top_level_var = NULL;
    matvar_t *matvar = NULL;
    top_level_var = Mat_VarReadNext(matfp);

    if (top_level_var == NULL || top_level_var->class_type != MAT_C_STRUCT) {
        printf("ERROR: MAT file does not contain a top-level struct variable as expected.\n");
        if (top_level_var != NULL) Mat_VarFree(top_level_var);
        Mat_Close(matfp);
        return 1;
    }

    matvar = Mat_VarGetStructFieldByName(top_level_var, (char*)"A", 0);

    if (matvar == NULL) {
        printf("ERROR: Could not find the required 'A' field inside the top-level structure.\n");
        Mat_Close(matfp);
        Mat_VarFree(top_level_var);
        return 1;
    }

    if (matvar->class_type != MAT_C_SPARSE) {
        printf("ERROR: Found field 'A' is not sparse (class type %d) in MAT file.\n", matvar->class_type);
        Mat_Close(matfp);
        Mat_VarFree(top_level_var);
        return 1;
    }

    int nrows = matvar->dims[0];
    int ncols = matvar->dims[1];

    mat_sparse_t *sparse_data = (mat_sparse_t*)matvar->data;

    int *ir_original = (int*)sparse_data->ir;
    int *jc_original = (int*)sparse_data->jc;
    if (!ir_original || !jc_original) {
        printf("Error: Sparse matrix data is NULL\n");
        Mat_VarFree(top_level_var);
        Mat_Close(matfp);
        return 1;
    }

    int nnz = (int)jc_original[ncols];

    int *rowptr = (int *)calloc(nrows + 1, sizeof(int));
    memErrchk(rowptr);
    int *index = (int *)malloc(nnz * sizeof(int));
    memErrchk(index);

    for (int j = 0; j < ncols; j++) {
        for (int p = jc_original[j]; p < jc_original[j+1]; p++) {
            int i = ir_original[p];
            if (i < nrows) {
                rowptr[i+1]++;
            }
        }
    }

    for (int i = 1; i <= nrows; i++) {
        rowptr[i] += rowptr[i-1];
    }

    int *temp_rowptr = (int *)malloc(nrows * sizeof(int));
    memErrchk(temp_rowptr);

    for (int i = 0; i < nrows; i++) {
        temp_rowptr[i] = rowptr[i];
    }

    for (int j = 0; j < ncols; j++) {
        for (int p = jc_original[j]; p < jc_original[j+1]; p++) {
            int i = ir_original[p];
            if (i < nrows) {
                int dest = temp_rowptr[i]++;
                index[dest] = j;
            }
        }
    }

    Mat_Close(matfp);
    free(temp_rowptr);
    Mat_VarFree(top_level_var);
// =============================================================

    int *labels = (int *)malloc(nrows*sizeof(int));
    memErrchk(labels);

    for (int i=0; i<nrows; i++) {   // Initialize labels
      labels[i] = i;
    }

    gettimeofday(&start, NULL);
    coloringCC_gpu_2(nrows, nnz, rowptr, index, labels);  // Call to coloringCC_gpu function. The elapsed time that is measured includes only the execution time of this function.
    gettimeofday(&end, NULL);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1e6;
    printf("Execution time: %f seconds\n", elapsed);


    printf("\nStarting validation on CPU...\n");
    qsort(labels, nrows, sizeof(int), compare_labels);
    int unique_components = 0;
    if (nrows > 0) {
        unique_components = 1;
        for (int i = 1; i < nrows; i++) {
            if (labels[i] != labels[i - 1]) {
            unique_components++;
            }
        }
    }

    printf("==========================================\n");
    printf("       CC VALIDATION RESULTS\n");
    printf("==========================================\n");
    printf(" Total Nodes processed  : %d\n", nrows);
    printf(" Connected Components   : %d\n", unique_components);
    printf("==========================================\n");


    free(index);free(rowptr);
    free(labels);
}

!nvcc -O3 -arch=sm_70 program.cu -lmatio -o program

!./program mawi_201512020330.mat
```

**An example of how the output should look like:**
```bash
rm -f executable
nvcc -03 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_75,code=sm_75  -I/home/d/dimopoul/local/matio/include -I/home/d/dimopoul/local/hdf5/include -I/home/d/dimopoul/local/zlib/include -o executable main_gpu.cu coloringCC_gpu.cu -L/home/d/dimopoul/local/matio/lib -L/home/d/dimopoul/local/hdf5/lib -L/home/d/dimopoul/local/zlib/lib -lhdf5 -lz -lmatio
job ID: 2242165
Running on node: cn22
CPUs per taks: 1
Copying /home/d/dimopoul/ergasia3_parallhla/matrices/europe_osm.mat to /scratch/d/dimopoul/2242165/europe_osm.mat
Staging complete. Starting execution...
Number of while iteration: 5324
Execution time: 1.142639 seconds

Starting validation on CPU...
==========================================
       CC VALIDATION RESULTS
==========================================
 Total Nodes processed  : 226196185
 Connected Components   : 3971144
==========================================
Cleaning up...
Removed /scratch/d/dimopoul/2243435
	
