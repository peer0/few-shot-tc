#!/bin/bash

for seed in 42; do
    for language in java python corcod; do
        for modelname in codebert graphcodebert unixcoder codet5p; do
            for lr in '1e-5'; do
                for aug in back-translation; do
                    echo "python3 superaug_main.py --config configs/${modelname}.json --aug $aug --lr $lr --seed $seed --dataset $language --checkpoint acc" 
                          python3 superaug_main.py --config configs/${modelname}.json --aug $aug --lr $lr --seed $seed --dataset $language --checkpoint acc \
                          > result_log/${language}_supervised_${modelname}_${aug}_${lr}_${seed}.log
                done
            done
        done
    done
done
