for nshot in 5; do
	for language in python ; do
		for pse_cl in 7; do
			for thres in  0.7; do
				for model in 'codet5p' ; do
					for lr in '45e-5' '5e-4' ; do
						echo "python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres}  ../data/problem_based_split/${language}_extended_data  ${pse_cl} > ../result_log_balance/${language}/${nshot}_${model}_${lr}_${thres}_${language}_${pse_cl}.log" 
						python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ../data/problem_based_split/${language}_extended_data ${pse_cl} > ../result_log_balance/${language}/${nshot}_${model}_${lr}_${thres}_${language}_${pse_cl}.log 			
					done
				done
			done
		done
	done
done




# for nshot in 10; do
# 	for language in java ; do
# 		for thres in  0.7; do
# 			for model in 'codet5p' ; do
# 				for lr in  '5e-4'  ; do
# 					echo "python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres}  ../data/problem_based_split/${language}_extended_data > ../result_log_balance/look/${nshot}_${model}_${lr}_${thres}_${language}.log" 
# 					python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ../data/problem_based_split/${language}_extended_data > ../result_log_balance/look/${nshot}_${model}_${lr}_${thres}_${language}.log 					
# 				done
# 			done
# 		done
# 	done
# done