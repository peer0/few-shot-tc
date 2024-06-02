for nshot in 10; do
	for language in corcod ; do
		for pse_cl in 5 ; do
			for thres in  0.7; do
				for max_epoch in 2; do 
					for model in 'codet5p' ; do
						for lr in  '1e-4' ; do
							for seed in 42 ; do   
								echo "python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ${max_epoch} ../data/problem_based_split/${language}_extended_data  ${pse_cl}"
                                python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ${max_epoch} ../data/problem_based_split/${language}_extended_data ${pse_cl} ${seed}
								#echo "python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ${max_epoch}  ../data/problem_based_split/${language}_extended_data  ${pse_cl}" 
								#python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ${max_epoch} ../data/problem_based_split/${language}_extended_data ${pse_cl}
							done
						done
					done
				done
			done
		done
	done
done