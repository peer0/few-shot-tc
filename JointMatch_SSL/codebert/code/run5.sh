#!/bin/bash
for nshot in 5; do
	for language in java python; do
		for thres in 0.8 0.9; do
			for model in 'codebert' 'unixcoder' 'codet5p'; do
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


# codet5p은 오류로 0.7이 안됨. 따로 돌려야함.