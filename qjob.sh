#!/bin/bash
#$ -N lonely
#$ -q ai.q@ahtapot-5-1
#$ -cwd
#$ -S /bin/bash
#$ -l gpu=1
##$ -l h_rt=48:00:00


source ../.bash_profile
julia lonelyrnn.jl --N 262144 --seqlength 7 > log_seqlength7.txt

rm lonely.*
