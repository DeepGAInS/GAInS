#!/bin/bash 
#SBATCH -J train
#SBATCH -p xhhgnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time 24:00:00
#SBATCH --comment=BASE
###SBATCH -o std.out.%j
###SBATCH -e std.err.%j
#SBATCH -o /work/home/acvwd4uw3y181/rsliu/detectron2/tools/std.out.%j
#SBATCH -e /work/home/acvwd4uw3y181/rsliu/detectron2/tools/std.err.%j


#########################################################
### Slurm scripts for Sugon Portal_5.0 of BASE
### Version 1.0    |  2019-09-09  |  created by Zhang Guoliang
### Version 1.0.1  |  2020-11-24  |  modified by Zhang Guoliang
#########################################################

### Get parameters from GUI

MIDFILE_DIR=/work/home/acvwd4uw3y181/rsliu/detectron2/tools/.portal
source $MIDFILE_DIR/job_portal.var
source $MIDFILE_DIR/job_interface.var

### Set basic var   ### MARK_slurm2pbs

JOBID=$SLURM_JOB_ID                                  ### slurm2pbs
NP=$SLURM_NPROCS                                     ### slurm2pbs
NNODE=`srun hostname | sort | uniq | wc -l`          ### slurm2pbs

LOG_FILE=$WORK_DIR/job_${JOB_NAME}_${JOBID}.log
HOST_FILE=$WORK_DIR/job_${JOB_NAME}_${JOBID}_${NP}c_${NNODE}n.ma 

srun hostname | sort | uniq -c |awk '{print $2":"$1}' > $HOST_FILE  ### slurm2pbs

### Write basic job infomations

echo -e "The start time is: `date +"%Y-%m-%d %H:%M:%S"` \n" | tee -a $LOG_FILE 
echo -e "My job ID is: $JOBID \n" | tee -a $LOG_FILE  
echo -e "The total cores is: $NP \n" | tee -a $LOG_FILE 
echo -e "The hosts is: \n" | tee -a $LOG_FILE
cat $HOST_FILE | tee -a $LOG_FILE
echo -e "\n"  | tee -a $LOG_FILE 

### Run APP

  # MARK_CMD  #Don't delete this line!!!
#!/bin/bash
## job script created by Gridview Jobmanager.
srun train_net.py --config-file /work/home/acvwd4uw3y181/rsliu/detectron2/tools/baseline_mask_rcnn_R_50_FPN_1x.yaml

  # MARK_BASH #Don't delete this line!!!

if [ $? -ne 0 ]; then
  exit 1
fi
echo The end time is: `date +"%Y-%m-%d %H:%M:%S"` | tee -a $LOG_FILE   

