#!/bin/bash

for seed in 42; do
    for nshot in 5; do
        for language in java; do
            for thres in 0.4; do
                for lr in '5e-4' ; do
                    echo "python3 revised_main_ssl.py --config configs/codet5p.json --n_labeled_per_class $nshot --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language" 
                    python3 revised_main_ssl.py --config configs/codet5p.json --n_labeled_per_class $nshot --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language \
                    > logs/codet5p_nshot${nshot}_thres${thres}_lr${lr}_seed${seed}_lang${language}_epoch20.log
                done
			done
		done
	done
done
