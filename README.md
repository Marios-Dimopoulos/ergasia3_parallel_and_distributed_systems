# Connected Components Labeling — CUDA (GPU)

University assignment (Exercise 3) for the **Parallel and Distributed
Systems** course: two CUDA (GPU) implementations of connected-components
labeling for large sparse graphs, benchmarked on the **Aristotle HPC
cluster** (Slurm, `gpu` partition, Tesla GPUs) and on Google Colab.

This is the third exercise in a series; the first (sequential / OpenMP
/ pthreads / OpenCilk) and second (hybrid MPI + OpenMP) implementations
of the same labeling algorithm live in separate repositories.

> **Note:** two of the three planned strategies are implemented here -
> one-thread-per-node and one-warp-per-node. A block-per-node strategy
> was planned but not completed in time.

## Files

| File                              | Purpose                                                     |
|-------------------------------------|-----------------------------------------------------------------|
| `main_gpu_TPN.cu`                   | Reads a `.mat` matrix via `matio`, builds CSR, runs + times + validates the TPN kernel |
| `main_gpu_WPN.cu`                   | Same, but for the WPN kernel                                     |
| `coloringCC_gpu_TPN.cu/.h`          | One-**t**hread-**p**er-**n**ode GPU labeling kernel               |
| `coloringCC_gpu_WPN.cu/.h`          | One-**w**arp-**p**er-**n**ode GPU labeling kernel                 |
| `Makefile`                          | Builds `executable_TPN` and `executable_WPN`                     |
| `bash_script_gpu_final_test_TPN.sh` | Slurm batch script for the TPN executable                        |
| `bash_script_gpu_final_test_WPN.sh` | Slurm batch script for the WPN executable                        |
| `numberOfCC.m` / `numberOfCC.sh`    | MATLAB script (+ its own Slurm job) that independently computes the ground-truth number of connected components, for validation |

## Algorithm

Both variants share the same two-kernel structure per while-iteration,
run until neither kernel produces any change:

- **Kernel 1 — label propagation.** Each node adopts the smallest label
  among itself and its CSR neighbors (same min-label rule as the CPU
  versions of this algorithm). This is where the two strategies differ:
  - **TPN (one thread per node):** each thread scans its node's entire
    neighbor list sequentially.
  - **WPN (one warp per node):** all 32 threads of a warp cooperatively
    scan one node's neighbor list (each thread takes every 32nd
    neighbor), then combine their partial results with a
    `__shfl_down_sync` register-level reduction. This balances load on
    "hub" nodes with very large neighbor counts, at the cost of using
    32x more threads per node than TPN.
- **Kernel 2 — pointer jumping (path compression).** Each node follows
  `labels[labels[idx]]`, roughly halving its remaining "distance" to the
  true minimum label per call. This gives near-logarithmic convergence
  on long chain-like structures (e.g. road networks) instead of the one
  extra hop per iteration that kernel 1 alone would provide. Calling it
  more than once per while-iteration was tested and found to increase
  total runtime despite further reducing the iteration count - kernel
  launch overhead outweighs the benefit.

Both kernels use `atomicExch` on the label write, even though each
thread/warp is the sole writer of its own node's label - the atomic
guards against benign read/write races when reading a neighbor's label
concurrently with another thread updating it (at worst, a few extra
iterations to converge; never an incorrect final result). It's kept for
clarity and because the extra cost is negligible on modern GPU
hardware.

An **early-return optimization** in kernel 1 skips the entire neighbor
scan once a node's label is already `0` (the global minimum). This
matters enormously on power-law graphs with a few very-high-degree
"hub" nodes (e.g. `mawi_201512020330.mat`): once a hub reaches label 0
(within a handful of iterations), every subsequent while-iteration would
otherwise still pay for scanning its entire (huge) neighbor list for no
benefit. Benchmarks show up to ~400x speedup from this check alone on
such graphs.

`long long` is used explicitly when computing thread/warp indices in
kernel 1 of both variants: in the WPN kernel, `blockIdx.x * blockDim.x`
can overflow a 32-bit `int` on large grids before the division by 32,
so 64-bit arithmetic is required there; the TPN kernel uses the same
cast for consistency, even though its grids stayed within `int` range
for the matrices tested here.

## Dependencies

Required software:
- `nvcc` (CUDA toolkit) - developed against CUDA 12.4
- The [`matio`](https://github.com/tbeu/matio) library, and its own
  dependencies: `zlib`, `HDF5`

**Note on the Aristotle cluster:** `matio` and its dependencies are not
available system-wide there, so they were built and installed locally
under:
```
$HOME/local/zlib
$HOME/local/hdf5
$HOME/local/matio
```
The code will not build or run unless these (or equivalent system-wide
installs) are available and correctly linked. The `Makefile` assumes
this `$HOME/local` layout for its include/library paths; adjust
`MATIO_BASE` in the `Makefile` if your installation lives elsewhere.

## Building

```bash
make          # builds both executable_TPN and executable_WPN
make clean    # removes them
```

`NVCCFLAGS` builds a "fat binary" (`-gencode` for `sm_60`/`sm_70`/
`sm_80`/`sm_75`) containing compiled code for several GPU architectures
at once; at run time the NVIDIA driver automatically selects the
matching one for whatever GPU is actually present.

## Running

### On a Slurm cluster (e.g. Aristotle)

1. Place your `.mat` file(s) in a `matrices/` directory.
2. Edit `INPUT_FILE_NAME` in whichever `bash_script_gpu_final_test_*.sh` you're using.
3. Submit:
   ```bash
   sbatch bash_script_gpu_final_test_TPN.sh   # or _WPN.sh
   ```

Each script rebuilds the project fresh (`make clean && make`), stages
the input matrix and the executable into a per-job scratch directory
(for faster I/O on large matrices), runs it, and cleans up the scratch
directory afterward regardless of how the job ends (success, failure,
or being killed - via a `trap ... EXIT` on a `cleanup()` function).
Output goes to `job_<job_id>.out`, errors to `job_<job_id>.err`.

### Locally (with the matio libraries installed)

```bash
export LD_LIBRARY_PATH=$HOME/local/matio/lib:$HOME/local/hdf5/lib:$HOME/local/zlib/lib:$LD_LIBRARY_PATH
./executable_TPN <path_to_matrix.mat>   # or ./executable_WPN
```

### Recommended: Google Colab

Setting up `matio` and its dependencies locally is fiddly, so **Google
Colab is the recommended way to try this code** - `matio` installs with
one `apt-get` line and there's a free GPU runtime available. In a Colab
notebook:

```
!apt-get update
!apt-get install -y libmatio-dev
!wget <url-to-a-.mat-file-from-the-SuiteSparse-Matrix-Collection>
```
Then write the combined source (`main_gpu_TPN.cu` + `coloringCC_gpu_TPN.cu`,
or the WPN equivalents) to a single file with `%%writefile program.cu`,
and in a final cell:
```
!nvcc -O3 -arch=sm_70 program.cu -lmatio -o program
!./program <matrix_filename>.mat
```
(Adjust `-arch` to match whatever GPU Colab assigns you.)

## Benchmark findings

Benchmarks used three matrices with very different structures:
`europe_osm.mat` (a European road network - low-degree, one giant
connected component, huge diameter), `kmer_V1r.mat` (also low-degree,
with shorter chain-like structures than europe_osm), and
`mawi_201512020330.mat` (a power-law / hub-dominated graph with small
diameter).

- **`THREADS_PER_BLOCK` sweet spot:** tested at 32/64/128/256/512/1024.
  Both extremes hurt performance - 32 threads/block creates far more
  blocks than the GPU's multiprocessors can schedule concurrently
  (wasting register/thread capacity before it's the limiting factor),
  while 1024 threads/block leaves too few blocks for effective latency
  hiding. **256** consistently gave the best results across all three
  test matrices.
- **Impact of kernel 2 (pointer jumping):** largest on `europe_osm.mat`
  (huge diameter, one giant component - chains benefit the most from
  path compression), smaller on `kmer_V1r.mat` (shorter chains, so
  kernel-launch overhead eats more of the benefit), and negligible on
  `mawi_201512020330.mat` (small diameter already, thanks to its hub
  structure).
- **Impact of the kernel-1 early-return:** roughly 2x speedup on
  `europe_osm` and `kmer_V1r`, but **up to ~400x** on
  `mawi_201512020330` - the hub nodes reach label 0 almost immediately,
  and without the early return every later iteration keeps re-scanning
  their enormous neighbor lists for nothing.
- **TPN vs. WPN:** WPN's per-warp load balancing helps dramatically on
  `mawi_201512020330` *without* the early-return optimization (~400s
  down to ~5s), but with early-return enabled, TPN still wins overall
  (~1.2s) - for this kind of hub-dominated, small-diameter graph,
  pruning redundant work (early return) beats balancing it (warp per
  node). On the two low-degree graphs, WPN is consistently *worse* than
  TPN: with 32 threads committed to every node regardless of its degree,
  a multiprocessor that could run 1024 TPN threads on 1024 different
  nodes in parallel can only run 1024 WPN threads on 32 nodes at a time.
- Raw benchmark data (used to produce the report's graphs) is kept in
  the `tests_launching_kernel_2_more_than_once`,
  `tests_showing_the_impact_of_early_return`, and
  `tests_using_combinations_of_threadsPerBlock_and_kernels` directories
  in the original repository.

## Notes / known limitations

- `nrows`/`ncols` are read as `int` from `matvar->dims`, and `ir`/`jc`
  sparse indices are cast to `int*` — this assumes matio was built with
  32-bit indices.
- Each executable reads and CSR-converts the `.mat` file on the host
  (CPU) before any GPU work begins; this file I/O and conversion time is
  excluded from the reported "Execution time" (which covers only the
  `coloringCC_gpu_*` call).
- `numberOfCC.m` / `numberOfCC.sh` independently compute the true number
  of connected components for a given matrix using MATLAB's `conncomp`,
  as a ground-truth cross-check against each kernel's own on-GPU
  validation output.

