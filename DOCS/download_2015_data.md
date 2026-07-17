1) Log in to CHAMP

2) Locate to scratch folder

> Command: cd /scratch/asheesh/scripts-graphcast/

3) Use proxy connection

*export http_proxy="http://14.xxx.xxx.xx.xx:xxxx"*

*export https_proxy="http://14. xxx.xxx.xx.xx:xxxx "*

*export ftp_proxy="ftp://14.xxx.xxx.xx.xx:xxxx "*

*export ftps_proxy="ftps://14.xxx.xxx.xx.xx:xxxx "*

4) Setup the environment

*source graphcast_env/bin/activate*

5) Run the download file

python data_collection/era5_downloader.py --year 2015 --region nagpur

6) python test_data_pipeline.py

7) python validate_training_inputs.py

8) written the PBS script:

```bash
#!/bin/bash

#PBS -N GraphCast_2015_Training
#PBS -q workq
#PBS -l select=1:ncpus=64:mpiprocs=1
#PBS -l place=scatter:excl
#PBS -l walltime=06:00:00
#PBS -V

echo "=========================================="
echo " GraphCast Training on CHAMP"
echo " Started: $(date)"
echo " Host: $(hostname)"
echo "=========================================="

cd /scratch/asheesh/scripts-graphcast || exit 1
source graphcast_env/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cpu

echo
echo "Python:"
which python
python --version

echo
echo "Modules:"
module list

echo
echo "=========================================="
echo "Stage 3 - Fine-tuning GraphCast on REAL ERA5 2015"
echo "=========================================="
time python run_unseen_workflow.py \
  --stage 3 \
  --year 2015 \
  --epochs 20 || exit 1

echo
echo "=========================================="
echo "Stage 2 - Validating Fine-tuned Model"
echo "=========================================="
time python run_unseen_workflow.py \
  --stage 2 \
  --year 2015 \
  --checkpoint checkpoints/model_2015.nc || exit 1

echo
echo "=========================================="
echo "Benchmark Hardware"
echo "=========================================="
time python benchmark_hardware.py

echo
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
```

9) qsub to submit the pbs script.

10) Jobid 483955.champ1

11) 484015.champ1
