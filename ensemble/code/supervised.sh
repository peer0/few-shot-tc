#!/bin/bash

for seed in 42; do
    for language in java python corcod; do
        for modelname in codebert graphcodebert unixcoder codet5p; do
            for lr in '1e-5'; do
                echo "python3 supervised_main.py --config configs/${modelname}.json --lr $lr --seed $seed --dataset $language --checkpoint acc" 
                python3 supervised_main.py --config configs/${modelname}.json $thres --lr $lr --seed $seed --dataset $language --checkpoint acc \
                > result_log/${language}_supervised_${modelname}_${lr}_${seed}.log
            done
        done
    done
done
