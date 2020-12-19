#!/bin/bash

data_folder="./$1/$2_results/cluster_data"
condor_folder="./$1/$2_results/cluster_data/condor_logging"

if [[ $# -ne 2 ]] ; then
    echo "Error: Must supply <objective {walker,branin2D,hart6D,micha10D}> <acquisition {EI,EIC,EIC_standard}>"
else

	echo "Total Arguments: " $#
	echo "(1) objective:   " $1
	echo "(2) acquisition: " $2

	echo "Removing data inside " ${data_folder} " ..."
	rm ${data_folder}/data_*
	echo "Removing data inside " ${data_folder} ".hydra/ ..."
	rm -r ${data_folder}/.hydra
	echo "Removing data inside " ${data_folder} " ..."
	rm ${condor_folder}/condor_*

	echo "Removing condor_XXX.XXX files inside ./ "
	rm ./condor_$2_*
fi
