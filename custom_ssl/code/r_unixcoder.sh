for nshot in 5; do
	for language in python ; do
		for pse_cl in 7; do
			for thres in  0.7; do
				for max_epoch in 100; do 				
					for model in 'unixcoder' ; do
						for lr in '45e-5' '5e-4' ; do
							echo "python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ${max_epoch}  ../data/problem_based_split/${language}_extended_data  ${pse_cl} > ../result_log_balance/${language}/${model}/class${pse_cl}/${nshot}_${model}_${lr}_${thres}_${language}_${pse_cl}.log" 
							python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ${max_epoch} ../data/problem_based_split/${language}_extended_data ${pse_cl} > ../result_log_balance/${language}/${model}/class${pse_cl}/${nshot}_${model}_${lr}_${thres}_${language}_${pse_cl}.log 			
						done
					done
				done
			done
		done
	done
done

