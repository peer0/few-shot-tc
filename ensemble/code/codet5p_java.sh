#!/bin/bash

for seed in 42 45; do
    for nshot in 10; do
        for language in java; do
            for modelname in codet5p; do
                for aug in natural; do
                    for thres in 0.7; do
                        for lr in 1e-5; do
                            echo "python3 revised_main_ssl.py --config configs/${modelname}.json --n_labeled_per_class $nshot --aug $aug --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language --checkpoint acc" 
                            python3 cross_main_ssl.py --config configs/${modelname}.json --n_labeled_per_class $nshot --aug $aug --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language --checkpoint acc \
                            > result_log/cross_${language}_${nshot}_${modelname}_${aug}_${lr}_${seed}.log
                        done
                    done
                done
            done
        done
    done
done

for seed in 42 45; do
    for nshot in 10; do
        for language in java; do
            for modelname in codet5p; do
                for aug in  natural; do
                    for thres in 0.7; do
                        for lr in 1e-5; do
                            echo "python3 cross_main_ssl.py --config configs/${modelname}.json --n_labeled_per_class $nshot --aug $aug --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language --checkpoint acc" 
                            python3 cross_original.py --config configs/${modelname}.json --n_labeled_per_class $nshot --aug $aug --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language --checkpoint acc \
                            > result_log/cross_${language}_${nshot}_${modelname}_${aug}_${lr}_${seed}.log
                        done
                    done
                done
            done
        done
    done
done
