#!/bin/bash

for seed in 42; do
    for nshot in 5 10; do
        for language in python; do
            for thres in 0.7; do
                for lr in '1e-4' '1e-5'; do
                    echo "python3 revised_main_ssl.py --config configs/codet5p.json --n_labeled_per_class $nshot --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language" 
                    python3 revised_main_ssl.py --config configs/codet5p.json --n_labeled_per_class $nshot --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language \
                    > logs/codet5p_nshot${nshot}_thres${thres}_lr${lr}_seed${seed}_lang${language}_epoch100.log 
                done
			done
		done
	done
done
5