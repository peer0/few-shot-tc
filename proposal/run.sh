#!/bin/bash

for trial_number in 2 3; do
	for language in java python; do
		for type in random problem; do
			python3 gpt.api.single.prediction.py --inputfile dataset/${language}.shinwoo.${type}.jsonl --model gemini-pro --filename ${language}.${type} --complexity_type single --tool shinwoo --index ${trial_number}
		done
	done
done
