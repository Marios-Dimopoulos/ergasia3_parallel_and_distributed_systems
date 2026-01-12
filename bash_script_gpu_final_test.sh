#!/bin/bash
#SBATCH --job-name=my_gpu_final_test_job
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:tesla:1
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --mem=20GB
#SBATCH --ntasks-per-node=1

module purge    
module load gcc/13.2.0-iqpfkya  # Load the necessary modules
module load cuda/12.4.0-zk32gam

make clean && make executable

export LD_LIBRARY_PATH=$HOME/local/matio/lib:$HOME/local/hdf5/lib:$HOME/local/zlib/lib:$LD_LIBRARY_PATH

INPUT_FILE_NAME="kmer_V1r.mat"  #<---- <---- <---- <---- Really important. Change this according to the matrix you want to test. The matric must be stored in the "matrices folder.""
SOURCE_FILE="$HOME/ergasia3_parallhla/matrices/$INPUT_FILE_NAME"
JOB_WORKING_DIR="/scratch/d/dimopoul/$SLURM_JOB_ID"

cleanup() {
    echo "Cleaning up..."
    if [ -n "$JOB_WORKING_DIR" ] && [ -d "$JOB_WORKING_DIR" ]; then # I set up a trap to clean up the scartch directory on whatever case. (Success, failure, killed etc.)
        rm -rf "$JOB_WORKING_DIR"
        echo "Removed $JOB_WORKING_DIR"
    fi
}

trap cleanup EXIT

mkdir -p "$JOB_WORKING_DIR"

echo "job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "Number of tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"

echo "Copying $SOURCE_FILE to $JOB_WORKING_DIR/$INPUT_FILE_NAME"    # Copy the input file to the scratch directory
/usr/bin/cp "$SOURCE_FILE" "$JOB_WORKING_DIR/$INPUT_FILE_NAME"

if [ $? -ne 0 ]; then
    echo "ERROR: File copy failed. Check source path or file size."
    exit 1
fi

/usr/bin/cp ./executable "$JOB_WORKING_DIR/executable"

echo "Staging complete. Starting execution..."

cd "$JOB_WORKING_DIR"

./executable "$JOB_WORKING_DIR/$INPUT_FILE_NAME"
