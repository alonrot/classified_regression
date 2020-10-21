#!/bin/bash
echo "Copying data from cluster ..."
echo "Total Arguments: " $#
echo "(1) objective:   " $1
echo "(2) acquisition: " $2

root_local="/Users/alonrot/MPI/WIP_projects/classified_regression/classireg/experiments/numerical_benchmarks"
root_remote="/home/amarcovalle/WIP_projects/classified_regression/classireg/experiments/numerical_benchmarks"

if [[ $# -ne 2 ]] ; then
    echo "Error: Must supply <objective {walker,branin2D,hart6D}> <acquisition {EI,EIC}>"
else
	echo "Creating folder " $root_local/$1/$2_results/cluster_data/ "..."
	mkdir -p $root_local/$1/$2_results/cluster_data/
	echo "Copying data ..."
	scp -r amarcovalle@login.cluster.is.localnet:$root_remote/$1/$2_results/cluster_data/* $root_local/$1/$2_results/cluster_data/
	
	echo "Creating folder " $root_local/$1/$2_results/cluster_data/.hydra "..."
	mkdir -p $root_local/$1/$2_results/cluster_data/.hydra
	echo "Copying data ..."
	scp -r amarcovalle@login.cluster.is.localnet:$root_remote/$1/$2_results/cluster_data/.hydra/* $root_local/$1/$2_results/cluster_data/.hydra/*
fi
