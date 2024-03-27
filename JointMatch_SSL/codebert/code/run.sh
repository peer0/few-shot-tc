#!/bin/bash

for nshot in 1 5 10; do
	for language in java python; do
		for thres in 0.7 0.8 0.9; do
			for model in 'microsoft/codebert-base' 'microsoft/unixcoder-base' 'Salesforce/codet5p-110m-embedding'; do
				for lr in '1e-5' '1e-4' '5e-4' ; do
					for tolerence in 10 20 30; do
						python3 panel_main_load.py ${nshot} ${model} ${lr} ${thres} ${tolerence} ../data/problem_based_split/${language}_extended_data > ../result_log/${nshot}_${model}_${lr}_${thres}_${tolerence}_${language}.log
					done
				done
			done
		done
	done
done
