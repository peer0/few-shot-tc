for nshot in 5; do
	for language in java corcod ; do
		for pse_cl in 5 6 7 ; do
			for thres in  0.7; do
				for max_epoch in 50; do 			
					for model in  'codesage' 'codebert' 'codet5p' 'unixcoder'; do
						for lr in '25e-5' '3e-4' '4e-4'; do #  '2e-4' '3e-4' '4e-4'
							echo "python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ${max_epoch}  ../data/problem_based_split/${language}_extended_data  ${pse_cl} > ../result_log_balance/${language}/${model}/class${pse_cl}/${nshot}_${model}_${lr}_${thres}_${language}_${pse_cl}.log" 
							python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ${max_epoch} ../data/problem_based_split/${language}_extended_data ${pse_cl} > ../result_log_balance/${language}/${model}/class${pse_cl}/${nshot}_${model}_${lr}_${thres}_${language}_${pse_cl}_${max_epoch}.log 		
						done
					done
				done
			done
		done
	done
done




# for nshot in 5 10; do
# 	for language in python ; do
# 		for pse_cl in 5 6 7 ; do
# 			for thres in  0.7; do
# 				for max_epoch in 50; do 			
# 					for model in  'graphcodebert' 'unixcoder'; do
# 						for lr in  '2e-4' '3e-4' '4e-4' ; do
# 							echo "python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ${max_epoch}  ../data/problem_based_split/${language}_extended_data  ${pse_cl} > ../result_log_balance/${language}/${model}/class${pse_cl}/${nshot}_${model}_${lr}_${thres}_${language}_${pse_cl}.log" 
# 							python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ${max_epoch} ../data/problem_based_split/${language}_extended_data ${pse_cl} > ../result_log_balance/${language}/${model}/class${pse_cl}/${nshot}_${model}_${lr}_${thres}_${language}_${pse_cl}_${max_epoch}.log 		
# 						done
# 					done
# 				done
# 			done
# 		done
# 	done
# done
