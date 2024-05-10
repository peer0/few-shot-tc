# #!/bin/bash

# for seed in 42 43 45; do
#     for nshot in 1 5 10; do
#         for language in java python corcod; do
#             for thres in 0.4 0.5 0.6; do
#                 for lr in '1e-5' '1e-4' '5e-4' ; do
#                     echo "python3 revised_main_ssl.py --config configs/codet5p.json --n_labeled_per_class $nshot --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language" 
#                     python3 revised_main_ssl.py --config configs/codet5p.json --n_labeled_per_class $nshot --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language > result_log/${language}_${nshot}_codet5p_${lr}_${thres}_${seed}.log
#                 done
# 			done
# 		done
# 	done
# done

#!/bin/bash

#!/bin/bash

for seed in 42; do
    for nshot in 5 10; do
        for language in python; do
            for thres in 0.7; do
                for lr in '1e-4' '1e-5'; do
                    echo "python3 revised_main_ssl2.py --config configs/codet5p.json --n_labeled_per_class $nshot --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language" 
                    python3 revised_main_ssl2.py --config configs/codet5p.json --n_labeled_per_class $nshot --psl_threshold_h $thres --lr $lr --seed $seed --dataset $language \
                    > logs/codet5p/v2/codet5p_nshot${nshot}_thres${thres}_lr${lr}_seed${seed}_lang${language}_epoch50_v2.log 
                done
			done
		done
	done
done

#!/bin/bash