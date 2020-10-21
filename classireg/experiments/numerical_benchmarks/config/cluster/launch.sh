#!/bin/bash

echo "Running ./config/cluster/launch.sh ..."
echo "Total Arguments:             " $#
echo "(1) repetition number:       " $1
echo "(2) max. number repetitions: " $2
echo "(3) algorithm:               " $3

export LD_LIBRARY_PATH=/lustre/home/amarcovalle/.mujoco/mujoco200/bin:/home/amarcovalle/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH

/home/amarcovalle/.anaconda3/envs/classireg/bin/python run_experiments.py run_type=individual rep_nr=$1 Nend=$2 acqui=$3