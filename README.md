This project implements the coloringCC labels algorithm using large sparse 
matrices. The input matrices are provided from Sutisparse collection repository,
in .mat format and are read using the matio library. The code was developed and 
evaluated on the Aristotle HPC cluster using Slurm.

DEPENDENCIES: Required software: NVCC, matio library, matio dependencies (zlib, HDF5)

DEPENDENCY INSTALLATION NOTE (IMPORTANT): On the Aristotle cluster, the matio library
and its dependencies were not available system-wide. Therefore, the following libraries
were installed by me: $HOME/local/zlib
		      $HOME/local/hdf5
		      $HOME/local/matio.
The code will not run unless these libraries are available and properly linked

For building the executable: make executable

This will compile "main_gpu.cu" and "coloringCC_gpu.cu" into the executable

Note: The Makefile assumes the matio library and its dependencies (HDF5, zlib)
are installed locally in $HOME/local. In the bash script i set the path needed.

Next up, in the commmand line type: sbatch bash_script_gpu_final_test.sh",
or whatever other slurm command for running a job, and everything happens
automatically. The .out and .err files are created. On the .out file is 
printed the output and on the .err file, the errors, if any happened to
appear. On The bash script, change the INPUT_FILE_NAME according to the 
file that you want the code to run on. Also, the example matrix, must be 
installed on a "matrices" directory because of the paths that i've set on 
the bash script. You can modify that script the way you like
