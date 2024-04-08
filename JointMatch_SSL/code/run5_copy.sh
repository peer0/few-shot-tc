for nshot in 10; do
	for language in java ; do
		for thres in  0.7; do
			for model in 'codet5p' ; do
				for lr in '8e-4' '9e-4' '65e-5' ; do
					echo "python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres}  ../data/problem_based_split/${language}_extended_data > ../result_log_tk512/${nshot}_${model}_${lr}_${thres}_${language}.log" 
					python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ../data/problem_based_split/${language}_extended_data > ../result_log_tk512/${nshot}_${model}_${lr}_${thres}_${language}.log 				
				done
			done
		done
	done
done