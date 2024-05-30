#!/bin/bash

for seed in 43; do
    for nshot in 5 10; do
        for language in python; do
            for thres in 0.7; do
                for lr in '4e-04' '3e-04'; do
                    echo "python3 revised_main_ssl.py --config configs/codesage.json --n_labeled_per_class $nshot --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language" 
                    python3 revised_main_ssl.py --config configs/codesage.json --n_labeled_per_class $nshot --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language \
                    > logs/codesage/${seed}/v1/codesage_nshot${nshot}_thres${thres}_lr${lr}_seed${seed}_lang${language}_epoch50.log 
                done
			done
		done
	done
done