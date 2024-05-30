#!/bin/bash

for seed in 42; do
    for nshot in 10; do
        for language in python; do
            for thres in 0.7; do
                for lr in '4e-5'; do
                    echo "python3 revised_main_ssl.py --config configs/codebert.json --n_labeled_per_class $nshot --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language" 
                    python3 revised_main_ssl.py --config configs/codebert.json --n_labeled_per_class $nshot --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language \
                    > logs/codebert/v1/${language}_${nshot}_codebert_${lr}_${thres}_${seed}_epoch50.log
                done
			done
		done
	done
done



for seed in 42; do
    for nshot in 5; do
        for language in python; do
            for thres in 0.7; do
                for lr in '1e-5'; do
                    echo "python3 revised_main_ssl2.py --config configs/codebert.json --n_labeled_per_class $nshot --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language" 
                    python3 revised_main_ssl2.py --config configs/codebert.json --n_labeled_per_class $nshot --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language \
                    > logs/codebert/v2/${language}_${nshot}_codebert_${lr}_${thres}_${seed}_epoch50.log
                done
			done
		done
	done
done
