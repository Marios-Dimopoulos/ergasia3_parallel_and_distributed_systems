# ColoringCC CUDA Implementation

This project implements the parallel Connected Components labeling algorithm using CUDA for large sparse matrices.

## Dependencies
* **NVCC** (CUDA Toolkit)
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
1. **Build:** `make executable`
2. **Setup:** Place your `.mat` files in a `/matrices` directory.
3. **Configure:** Edit `INPUT_FILE_NAME` in `bash_script_gpu_final_test.sh`
4. **Submit:** `sbatch bash_script_gpu_final_test.sh`
5. **Output:** On the .out file
6. **Errors:** on the .err file

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
Cleaning up...
Removed /scratch/d/dimopoul/2242165

	
	
