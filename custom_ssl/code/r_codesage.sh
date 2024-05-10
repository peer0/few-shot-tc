for nshot in 5 10; do
	for language in python ; do
		for pse_cl in 1 7 ; do
			for thres in  0.7; do
				for max_epoch in 50; do 			
					for model in 'codebert' 'codesage' 'codet5p' 'graphcodebert' 'unixcoder'; do
						for lr in  '2e-4' '3e-4' '4e-4' ; do
							echo "python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ${max_epoch}  ../data/problem_based_split/${language}_extended_data  ${pse_cl} > ../result_log_balance/${language}/${model}/class${pse_cl}/${nshot}_${model}_${lr}_${thres}_${language}_${pse_cl}.log" 
							python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ${max_epoch} ../data/problem_based_split/${language}_extended_data ${pse_cl} > ../result_log_balance/${language}/${model}/class${pse_cl}/${nshot}_${model}_${lr}_${thres}_${language}_${pse_cl}.log 			
						done
					done
				done
			done
		done
	done
done
