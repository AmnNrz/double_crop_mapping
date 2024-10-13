#!/bin/bash
cd /home/a.norouzikandelati/Projects/Double_Crop_Mapping/qsubs

# Run all 150 jobs
batch_number=121
while [ $batch_number -le 150 ]
do
  sbatch ./q_Jeol_$batch_number.sh
  let "batch_number+=1"
done