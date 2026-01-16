#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <matio.h>
#include <cuda_runtime.h>

#include "coloringCC_gpu_WPN.h"

#define memErrchk(ptr) { memAssert((ptr), __FILE__, __LINE__); }
inline void memAssert(void *ptr, const char *file, int line, bool abort=true) { // Macro for error checking memory allocations. Makes the code more readable and easier to debug.
    if (ptr == NULL) {
        fprintf(stderr, "Memory Allocation Failed at: %s:%d\n", file, line);    
        if (abort) exit(EXIT_FAILURE);                                          
    }                                                                           
}

int compare_labels(const void *a, const void *b) {  // I use this function qsort function, to get a more efficient validation of the results.
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
    coloringCC_gpu_WPN(nrows, nnz, rowptr, index, labels);  // Call to coloringCC_gpu function. The elapsed time that is measured includes only the execution time of this function.
    gettimeofday(&end, NULL);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1e6;
    printf("Execution time: %f seconds\n", elapsed);
// ========== Validation of results ==========
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
// ============================================
    free(index);free(rowptr);
    free(labels);
}
