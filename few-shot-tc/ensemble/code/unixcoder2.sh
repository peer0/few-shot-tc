#!/bin/bash

echo "### START DATE=$(date)" 
echo "### HOSTNAME=$(hostname)" 
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" 

for seed in 43; do
    for nshot in 5 10; do
        for language in python; do
            for thres in 0.7; do
                for model in 'unixcoder'; do
                    for lr in '1e-04' '4e-05'; do
                        for aug in 'forwhile' 'back-translation'; do
                            for version in 'v2'; do
                                log_path="logs/${model}/${seed}/${aug}/${version}/${model}_nshot${nshot}_thres${thres}_lr${lr}_seed${seed}_lang${language}_epoch50_${version}.log"
                                log_dir=$(dirname ${log_path})
                                mkdir -p ${log_dir}
                                echo "python revised_main_ssl.py --config configs/${model}.json --n_labeled_per_class $nshot --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language" 
                                python revised_main_ssl.py --config configs/${model}.json --n_labeled_per_class $nshot --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language --aug $aug --version $version\
                                > ${log_path}
                            done
                        done
                    done
                done
            done
        done
    done
done


echo "###" 
echo "### END DATE=$(date)"