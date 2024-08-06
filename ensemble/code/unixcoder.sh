#!/bin/bash

for seed in 42; do
    for language in  java; do
        for nshot in 10; do
            for modelname in unixcoder; do
                for aug in none; do
                    for thres in 0.8; do
                        for lr in 1e-5; do
                            echo "python3 revised_main_ssl.py --config configs/${modelname}.json --n_labeled_per_class $nshot --aug $aug --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language --checkpoint acc" 
                            script -q -c "python3 revised_main_ssl.py --config configs/${modelname}.json --n_labeled_per_class $nshot --aug $aug --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language --checkpoint acc" \
                            -a result_log/cross_${language}_${nshot}_${modelname}_${aug}_${lr}_${seed}.log
                        done
                    done
                done
            done
        done
    done
done
