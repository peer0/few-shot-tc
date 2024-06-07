#!/bin/bash
echo "### START DATE=$(date)" 
echo "### HOSTNAME=$(hostname)" 
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" 

for seed in 42 43 45; do
    for nshot in 5 10; do
        for language in java; do
            for thres in 0.7; do
                for model in 'codebert'; do
                    for lr in '1e-05' '2e-06'; do
                        for aug in 'forwhile'; do
                            for version in 'v1' 'v2' 'v3' 'v4'; do
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
                                > ${log_path}
                            done
                        done
                    done
                done
            done
        done
    done
done


for seed in 42 43 45; do
    for nshot in 5 10; do
        for language in java; do
            for thres in 0.7; do
                for model in 'codebert'; do
                    for lr in '1e-05' '2e-06'; do
                        for aug in 'back-translation'; do
                            for version in 'v1' 'v3' 'v4'; do
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

