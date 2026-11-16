#!/bin/bash
# Benchmark complet CUDA vs OpenMP target sur un GPU GH200 de ROMEO.
# Lancement : cd ~/benchmark-omp-cuda-cg && sbatch slurm/bench.slurm
#SBATCH -J cg-omp-cuda
#SBATCH -p short
#SBATCH -A r260073
#SBATCH --constraint=armgpu
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o results/bench_%j.log

set -u
module load nvhpc/nvhpc-nompi/24.11
mkdir -p results
OUT=results/raw_${SLURM_JOB_ID}.csv
REPS=5
echo "version,matrix,m,n,nnz,test,rep,time_s,gbps,iters,relres,true_relres,err_inf" > $OUT

# contexte de la mesure (pour le README)
{ hostname; nvidia-smi --query-gpu=name,memory.total,clocks.max.memory,driver_version --format=csv;
  nvc++ --version | head -2; nvcc --version | tail -2; } > results/env_${SLURM_JOB_ID}.txt

./bin/cg_cuda stream $REPS >> $OUT

CASES="poisson2d:512 poisson2d:1024 poisson2d:2048 poisson2d:4096 poisson3d:64 poisson3d:128 poisson3d:256 varcoef2d:1024 varcoef2d:2048"
for c in $CASES; do
  mat=${c%%:*}; m=${c##*:}
  for v in cuda cuda-fused cusparse cuda-managed cuda-system; do ./bin/cg_cuda $v $mat $m $REPS all >> $OUT; done
  for v in omp-naive omp-devptr omp-loop omp-opt omp-fused; do ./bin/cg_omp $v $mat $m $REPS all >> $OUT; done
  ./bin/cg_omp_um omp-unified $mat $m $REPS all >> $OUT
  echo "fini : $c $(date +%T)"
done

# balayage de thread_limit pour omp-opt (noyaux seulement)
SW=results/sweep_${SLURM_JOB_ID}.csv
echo "tl,version,matrix,m,n,nnz,test,rep,time_s,gbps,iters,relres,true_relres,err_inf" > $SW
for tl in 64 128 256 512 1024; do
  OMP_TL=$tl ./bin/cg_omp omp-opt poisson2d 2048 $REPS kernels | sed "s/^/$tl,/" >> $SW
done
echo "termine $(date)"
