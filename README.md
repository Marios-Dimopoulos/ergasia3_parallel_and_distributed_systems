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

  I recommend running the program on the google colab, because it's the easiest way.
  You don't need to set the matio libray and its dependencies manually. At the very top cell you just
  type the commands: `!apt-get update \ !apt-get install -y libmatio-dev`, and everything about the matio
  library is set. Then you'll need to download the graph using wget... so here there is no need for 
  creating the "matrices" directory. On the next cell, at the very start of it you type
  `%%writefile program.cu` and below you copy the code of main_gpu_TPN.cu and coloringCC_gpu_TPN.cu
  or main_gpu_WPN.cu and coloringCC_gpu_WPN.cu. At the last cell, you just type:
  `!nvcc -O3 -arch=sm_70 program.cu -lmatio program` and at the end you type:
  `!./program (the name of the graph)` and it's done. On the github repository there will be a 
  file called: colab_code, and there will be an example of how the colab version should be.
  
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
	
