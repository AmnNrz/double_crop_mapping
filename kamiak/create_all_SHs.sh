#!/bin/bash
cd /home/a.norouzikandelati/Projects/Double_Crop_Mapping/

block_count=100
batch_number=1
while [ $batch_number -le 100 ]
do
  cp template.sh    ./qsubs/q_Jeol_$batch_number.sh
  sed -i s/block_count/"$block_count"/g    ./qsubs/q_Jeol_$batch_number.sh
  sed -i s/batch_number/"$batch_number"/g  ./qsubs/q_Jeol_$batch_number.sh
  let "batch_number+=1" 
done