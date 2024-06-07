#!/bin/bash
echo "### START DATE=$(date)" 
echo "### HOSTNAME=$(hostname)" 
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" 

for seed in 45; do
    for nshot in 5; do
        for language in java; do
            for thres in 0.7; do
                for model in 'codet5p'; do
                    for lr in '1e-04'; do
                        for aug in 'forwhile'; do
                            for version in 'v3'; do
                                log_path="logs/${model}/${seed}/${aug}/${version}/${model}_nshot${nshot}_thres${thres}_lr${lr}_seed${seed}_lang${language}_epoch50_${version}.log"
                                log_dir=$(dirname ${log_path})
                                mkdir -p ${log_dir}
                                echo "python revised_main_ssl.py --config configs/${model}.json --n_labeled_per_class $nshot --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language" 
                                python revised_main_ssl.py \
                                --config configs/${model}.json \
                                --n_labeled_per_class $nshot \
                                --psl_threshold_h $thres \
                                --lr $lr \
                                --seed $seed \
                                --dataset $language \
                                --aug $aug \
                                --version $version\
                                --model $model \
                                --seed $seed 
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

