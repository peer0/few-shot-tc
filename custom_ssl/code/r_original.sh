for nshot in 5 10 ; do
	for language in python ; do
		for thres in  0.7; do
			for max_epoch in 37; do 
				for model_list in 'codet5p,codet5p' 'unixcoder,unixcoder' ; do
					if [ "$model_list" == "codet5p,codet5p" ]; then
						lr=("3e-4 2.5e-4")
					elif [ "$model_list" == "unixcoder,unixcoder" ]; then
						lr=("4e-4 2e-4")
					fi
					for seed in 42 43 45 ; do   
						echo "python3 panel_main_original.py ${nshot} '${model_list}' '${lr}' ${thres} ${max_epoch} ../data/jointmatch/${language} ${seed} > ./jointmatch_log/${language}/${nshot}_${model_list}_${thres}_${language}_${max_epoch}_${seed}.log"
						python3 panel_main_original.py ${nshot} "${model_list}" "${lr}" ${thres} ${max_epoch} ../data/jointmatch/${language} ${seed} #> ./jointmatch_log/${language}/${nshot}_${model_list}_${thres}_${language}_${max_epoch}_${seed}.log
					done
				done
			done
		done
	done
done






# #for nshot in 5 10; do
# for nshot in 5; do
# 	for language in python ; do
# 		for thres in  0.7; do
# 			for max_epoch in 2; do 
# 				#for model_list in 'codet5p,codet5p' 'unixcoder,unixcoder' 'codet5p,unixcoder'; do	
# 				for model_list in 'codet5p,codet5p' 'unixcoder,unixcoder' ; do
# 					for lr in  "3e-4 2.5e-4" "4e-4 2e-4" ; do
# 						for seed in 42 43 45 ; do   
# 							echo "python3 panel_main_original.py ${nshot} '${model_list}' '${lr}' ${thres} ${max_epoch} ../data/jointmatch/${language} ${seed} > ./jointmatch_log/${language}/${nshot}_${model_list}_${thres}_${language}_${max_epoch}_${seed}.log"
#                             python3 panel_main_original.py ${nshot} "${model_list}" "${lr}" ${thres} ${max_epoch} ../data/jointmatch/${language} ${seed} > ./jointmatch_log/${language}/${nshot}_${model_list}_${thres}_${language}_${max_epoch}_${seed}.log
# 						done
# 					done
# 				done
# 			done
# 		done
# 	done
# done











# for nshot in 5; do
#     for language in python; do
#         for thres in 0.7; do
#             for max_epoch in 2; do
#                 for model_list in 'codet5p,unixcoder'; do
#                     for lr in "1e-4 2e-4" "2e-4 5e-5"; do
#                         for seed in 42; do
#                             echo "python3 panel_main_original.py ${nshot} '${model_list}' '${lr}' ${thres} ${max_epoch} ../data/jointmatch/${language} ${seed} > ./jointmatch_log/${language}/${nshot}_${model_list}_${thres}_${language}_${max_epoch}_${seed}.log"
#                             python3 panel_main_original.py ${nshot} "${model_list}" "${lr}" ${thres} ${max_epoch} ../data/jointmatch/${language} ${seed} #> ./jointmatch_log/${language}/${nshot}_${model_list}_${thres}_${language}_${max_epoch}_${seed}.log
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done






# for nshot in 5 10; do
# 	for language in python ; do
# 		#for pse_cl in 5 ; do
# 			for thres in  0.7; do
# 				for max_epoch in 37; do 
# 					for model in 'codesage' 'codebert' 'codet5p' 'graphcodebert' 'unixcoder' ; do
# 						#for lr in  '1e-4 2e-4' ; do
# 							for seed in 42 43 45 ; do   
#  								echo "python3 panel_main_original.py ${nshot} ${model} ${thres} ${max_epoch}  ../data/jointmatch/${language} ${seed}> ./jointmatch_log/${language}/${nshot}_${model}_${thres}_${language}_${max_epoch}_${seed}.log" 
#                                 python3 panel_main_original.py ${nshot} ${model} ${thres} ${max_epoch}  ../data/jointmatch/${language} ${seed}> ./jointmatch_log/${language}/${nshot}_${model}_${thres}_${language}_${max_epoch}_${seed}.log 
								
								
# 								#echo "python3 panel_main_original.py ${nshot} ${model} ${lr} ${thres} ${max_epoch} ../data/problem_based_split/${language}_extended_data  ${pse_cl}"
#                                 #python3 panel_main_original.py ${nshot} ${model} ${lr} ${thres} ${max_epoch} ../data/problem_based_split/${language}_extended_data ${pse_cl} ${seed}
								
# 							done
# 						#done
# 					done
# 				done
# 			done
# 		#done
# 	done
# done
