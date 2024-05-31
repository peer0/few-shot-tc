for nshot in 1; do
	for language in python ; do
		for pse_cl in 1 ; do
			for thres in  0; do
				for max_epoch in 2; do 
					for model in 'codebert' ; do
						for lr in  '1e-4' ; do
							for seed in 42 43 45; do   
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