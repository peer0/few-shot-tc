#!/bin/bash

for seed in 42; do
    for nshot in 5 10 20; do
        #for language in java python corcod; do
        for language in corcod; do
            for modelname in unixcoder; do
                for thres in 0.8; do
                    for lr in '1e-5'; do
                        echo "python3 revised_main_ssl.py --config configs/${modelname}.json --n_labeled_per_class $nshot --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language --checkpoint acc" 
                        python3 revised_main_ssl.py --config configs/${modelname}.json --n_labeled_per_class $nshot --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language --checkpoint acc \
                        > result_log/${language}_${nshot}_${modelname}_${lr}_${seed}.log
                    done
                done
            done
        done
    done
done
