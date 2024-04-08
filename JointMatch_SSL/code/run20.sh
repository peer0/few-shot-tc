for nshot in 20 30 40 ; do
	for language in java ; do
		for thres in  0.7; do
			for model in 'codet5p' ; do
				for lr in '5e-4' '6e-4' ; do
					echo "python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres}  ../data/problem_based_split/${language}_extended_data > ../result_log_tk512/${nshot}_${model}_${lr}_${thres}_${language}.log" 
					python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ../data/problem_based_split/${language}_extended_data > ../result_log_tk512/${nshot}_${model}_${lr}_${thres}_${language}.log 				
				done
			done
		done
	done
done
