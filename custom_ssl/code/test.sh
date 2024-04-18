for nshot in 10; do
	for language in python ; do
		for pse_cl in 4; do
			for thres in  0.3; do
				for model in 'graphcodebert' ; do
					for lr in  '5e-4' ; do
						echo "python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres}  ../data/problem_based_split/${language}_extended_data  ${pse_cl}" 
						python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ../data/problem_based_split/${language}_extended_data ${pse_cl}
					done
				done
			done
		done
	done
done