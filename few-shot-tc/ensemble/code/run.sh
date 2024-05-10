#!/bin/bash
#test용
for seed in 42; do
    for nshot in 5; do
        for language in python; do
            for thres in 0.7; do
                for lr in '1e-4'; do
                    echo "python3 revised_main_ssl.py --config configs/codet5p.json --n_labeled_per_class $nshot --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language" 
                    python3 revised_main_ssl2.py --config configs/codet5p.json --n_labeled_per_class $nshot --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language
                done
			done
		done
	done
done
5