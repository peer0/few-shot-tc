#!/bin/bash

for seed in 42; do
    for nshot in 5 10; do
        for language in python; do
            for thres in 0.7; do
                for lr in '1e-5' '4e-05' '1e-4' '5e-4'; do
                    echo "python3 revised_main_ssl2.py --config configs/codebert.json --n_labeled_per_class $nshot --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language" 
                    python3 revised_main_ssl2.py --config configs/codebert.json --n_labeled_per_class $nshot --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language \
                    > logs/codebert/v2/${language}_${nshot}_codebert_${lr}_${thres}_${seed}_epoch50_v2.log
                done
			done
		done
	done
done
