
for nshot in 10; do
	for language in python; do
		for thres in  0.7 0.8 0.9; do
			for model in 'codebert' 'unixcoder' 'codet5p' ; do
				for lr in '1e-5' '1e-4' '5e-4' ; do
					for tolerence in 10 20 30; do
						echo "python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ${tolerence} ../data/problem_based_split/${language}_extended_data > ../result_log/${nshot}_${model}_${lr}_${thres}_${tolerence}_${language}.log" 
						python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ${tolerence} ../data/problem_based_split/${language}_extended_data > ../result_log/${nshot}_${model}_${lr}_${thres}_${tolerence}_${language}.log 

					done
				done
			done
		done
	done
done

# 10_codebert - 1e-5 0.7 20 오류뜸

#panel_main_load.py 10 unixcoder 1e-4 0.8 20 오류