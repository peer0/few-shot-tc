#!/bin/bash

for seed in 42; do
    for aug in artificial natural; do
        for language in corcod python java; do
            for nshot in 5 10; do
                for modelname in unixcoder; do
                    for thres in 0.8; do
                        for lr in 5e-6; do
                            echo "python3 cross_original.py --config configs/${modelname}.json --n_labeled_per_class $nshot --aug $aug --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language --checkpoint acc" 
                            python3 cross_original.py --config configs/${modelname}.json --n_labeled_per_class $nshot --aug $aug --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language --checkpoint acc \
                            > result_log/cross_${language}_${nshot}_${modelname}_${aug}_${lr}_${seed}.log
                        done
                    done
                done
            done
        done
    done
done
