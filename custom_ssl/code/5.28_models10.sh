for nshot in 10; do
    for language in java ; do
        for pse_cl in 7 ; do
            for thres in  0.7; do
                for max_epoch in 37; do             
                    for model in  'codesage' 'codebert' 'codet5p' 'graphcodebert' 'unixcoder' ; do
                        for seed in 42 43 45; do
                            if [ "${model}" = "codesage" ]; then
                                lr_values='3e-4'  # Two learning rates for codesage
                            elif [ "${model}" = "codebert" ]; then
                                lr_values='2e-4'  # One learning rate for codebert
                            elif [ "${model}" = "codet5p" ]; then
                                lr_values='2.5e-4'  # One learning rate for codet5p
                            elif [ "${model}" = "graphcodebert" ]; then
                                lr_values='2e-4'  # One learning rate for graphcodebert
                            elif [ "${model}" = "unixcoder" ]; then
                                lr_values='2e-4'  # One learning rate for unixcoder
                            fi

                            for lr in $lr_values ; do
                                echo "python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ${max_epoch} ../data/problem_based_split/${language}_extended_data  ${pse_cl} > ../result_log_balance/${language}/${model}/class${pse_cl}/${nshot}_${model}_${lr}_${thres}_${language}_${pse_cl}_${max_epoch}_${seed}.log" 
                                python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ${max_epoch} ../data/problem_based_split/${language}_extended_data ${pse_cl} ${seed}> ../result_log_balance/${language}/${model}/class${pse_cl}/${nshot}_${model}_${lr}_${thres}_${language}_${pse_cl}_${max_epoch}_${seed}.log         
                            done
                        done
                    done
                done
            done
        done
    done
done


# #for nshot in 5 10; do
# for nshot in 10; do
#     if [ "${nshot}" = "10" ]; then
#         #languages='python java corcod'
#         languages='python'
# 		models='codesage codebert codet5p graphcodebert unixcoder'
#         for language in $languages; do
#             if [ "${language}" = "corcod" ]; then
#                 pse_cls='5'
#             else
#                 pse_cls='7'
#             fi
#             for pse_cl in $pse_cls; do
#                 for thres in 0.7; do
#                     for max_epoch in 37; do
#                         for model in $models; do
#                             for seed in 42 43 45; do
#                                 if [ "${model}" = "codesage" ]; then
#                                     lr_values='3e-4'
#                                 elif [ "${model}" = "codebert" ]; then
#                                     lr_values='2e-4'
#                                 elif [ "${model}" = "codet5p" ]; then
#                                     lr_values='2.5e-4'
#                                 elif [ "${model}" = "graphcodebert" ]; then
#                                     lr_values='2e-4'
#                                 elif [ "${model}" = "unixcoder" ]; then
#                                     lr_values='2e-4'
#                                 fi

#                                 for lr in $lr_values; do
#                                     echo "python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ${max_epoch} ../data/problem_based_split/${language}_extended_data ${pse_cl} ${seed} > ../result_log_balance/${language}/${model}/class${pse_cl}/${nshot}_${model}_${lr}_${thres}_${language}_${pse_cl}_${max_epoch}_${seed}.log"
#                                     python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ${max_epoch} ../data/problem_based_split/${language}_extended_data ${pse_cl} ${seed} > ../result_log_balance/${language}/${model}/class${pse_cl}/${nshot}_${model}_${lr}_${thres}_${language}_${pse_cl}_${max_epoch}_${seed}.log
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     else
#         #languages='python java corcod'
#         languages='python'
#         models='codesage codebert codet5p graphcodebert unixcoder'
#         for language in $languages; do
#             if [ "${language}" = "corcod" ]; then
#                 pse_cls='5'
#             else
#                 pse_cls='7'
#             fi
#             for pse_cl in $pse_cls; do
#                 for thres in 0.7; do
#                     for max_epoch in 37; do
#                         for model in $models; do
#                             for seed in 42 43 45; do   
#                                 if [ "${model}" = "codesage" ]; then
#                                     lr_values='4e-4'  # Only two learning rates for codesage
#                                 elif [ "${model}" = "codebert" ]; then
#                                     lr_values='4e-4'
#                                 elif [ "${model}" = "codet5p" ]; then
#                                     lr_values='3e-4'
#                                 elif [ "${model}" = "graphcodebert" ]; then
#                                     lr_values='2.5e-4'
#                                 elif [ "${model}" = "unixcoder" ]; then
#                                     lr_values='4e-4'
#                                 fi

#                                 for lr in $lr_values; do
#                                     echo "python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ${max_epoch} ../data/problem_based_split/${language}_extended_data ${pse_cl} ${seed} > ../result_log_balance/${language}/${model}/class${pse_cl}/${nshot}_${model}_${lr}_${thres}_${language}_${pse_cl}_${max_epoch}_${seed}.log"
#                                     python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ${max_epoch} ../data/problem_based_split/${language}_extended_data ${pse_cl} ${seed} > ../result_log_balance/${language}/${model}/class${pse_cl}/${nshot}_${model}_${lr}_${thres}_${language}_${pse_cl}_${max_epoch}_${seed}.log
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     fi
# done



# for nshot in 10; do
#     for language in corcod ; do
#         for pse_cl in 5 ; do
#             for thres in  0.7; do
#                 for max_epoch in 37; do             
#                     for model in 'codesage' 'codebert' 'codet5p' 'graphcodebert' 'unixcoder'; do
#                         for seed in 42 43 45; do
#                             if [ "${model}" = "codesage" ]; then
#                                 lr_values='3e-4'  # Two learning rates for codesage
#                             elif [ "${model}" = "codebert" ]; then
#                                 lr_values='2e-4'  # One learning rate for codebert
#                             elif [ "${model}" = "codet5p" ]; then
#                                 lr_values='2.5e-4'  # One learning rate for codet5p
#                             elif [ "${model}" = "graphcodebert" ]; then
#                                 lr_values='2e-4'  # One learning rate for graphcodebert
#                             elif [ "${model}" = "unixcoder" ]; then
#                                 lr_values='2e-4'  # One learning rate for unixcoder
#                             fi

#                             for lr in $lr_values ; do
#                                 echo "python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ${max_epoch} ../data/problem_based_split/${language}_extended_data  ${pse_cl} > ../result_log_balance/${language}/${model}/class${pse_cl}/${nshot}_${model}_${lr}_${thres}_${language}_${pse_cl}_${max_epoch}_${seed}.log" 
#                                 python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ${max_epoch} ../data/problem_based_split/${language}_extended_data ${pse_cl} ${seed}> ../result_log_balance/${language}/${model}/class${pse_cl}/${nshot}_${model}_${lr}_${thres}_${language}_${pse_cl}_${max_epoch}_${seed}.log         
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done




#
# for nshot in 5 ; do
#     for language in corcod ; do
#         for pse_cl in 7 ; do
#             for thres in  0.7; do
#                 for max_epoch in 37; do             
#                     for model in 'codesage' 'codebert' 'codet5p' 'graphcodebert' 'unixcoder'; do
# 					#for model in 'codesage' 'codebert' 'codet5p' 'graphcodebert' 'unixcoder'; do
#                         for seed in 42 43 45; do   
# 							if [ "${model}" = "codesage" ]; then
# 								lr_values='4e-4'  # Only two learning rates for codesage
# 							elif [ "${model}" = "codebert" ]; then
# 								lr_values='4e-4'
# 							elif [ "${model}" = "codet5p" ]; then
# 								lr_values='3e-4'
# 							elif [ "${model}" = "graphcodebert" ]; then
# 								lr_values='2.5e-4'
# 							elif [ "${model}" = "unixcoder" ]; then
# 								lr_values='4e-4'
# 							fi

#                         	for lr in $lr_values ; do
#                             	echo "python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ${max_epoch} ../data/problem_based_split/${language}_extended_data  ${pse_cl} > ../result_log_balance/${language}/${model}/class${pse_cl}/${nshot}_${model}_${lr}_${thres}_${language}_${pse_cl}_${max_epoch}_${seed}.log" 
#                                 python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ${max_epoch} ../data/problem_based_split/${language}_extended_data ${pse_cl} ${seed}> ../result_log_balance/${language}/${model}/class${pse_cl}/${nshot}_${model}_${lr}_${thres}_${language}_${pse_cl}_${max_epoch}_${seed}.log         
# 							done
# 						done
#                     done
#                 done
#             done
#         done
#     done
# done

