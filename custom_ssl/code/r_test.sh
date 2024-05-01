for nshot in 1; do
	for language in python ; do
		for pse_cl in 1; do
			for thres in  0; do
				for max_epoch in 5; do 
					for model in 'codebert' ; do
						for lr in  '1e-4' ; do
							echo "python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ${max_epoch}  ../data/problem_based_split/${language}_extended_data  ${pse_cl}" 
							python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ${max_epoch} ../data/problem_based_split/${language}_extended_data ${pse_cl}
						done
					done
				done
			done
		done
	done
done



#'ast-t5'

