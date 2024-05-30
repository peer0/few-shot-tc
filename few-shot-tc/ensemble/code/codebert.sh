#!/bin/bash

for seed in 43; do
    for nshot in 5 10 20; do
        for language in python; do
            for thres in 0.7; do
                for lr in '2e-4'; do
                    echo "python3 revised_main_ssl.py --config configs/codebert.json --n_labeled_per_class $nshot --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language" 
                    python3 revised_main_ssl.py --config configs/codebert.json --n_labeled_per_class $nshot --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language \
                    > logs/codebert/${seed}/v1/${language}_${nshot}_codebert_${lr}_${thres}_${seed}_epoch50.log
                done
			done
		done
	done
done
