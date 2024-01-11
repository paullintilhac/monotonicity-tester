#!/bin/bash

#train robust combine three
time python train_combine_monotonic.py --train robustness_spec/seed_train_malicious/mutate_insert_any_pt1/pickles --model_name robust_combine_three >! ../models/adv_trained/robust_combine_three.log 2>&1&
#train monotonic
python monotonic.py --num_trees 1000 --model_name model_1000learner
#train robust monotonic
time python train_insert_monotonic.py --model_name robust_monotonic
#train_adv_combine
time python train_adv_combine.py --batch_size 50 --batches 132900 --verbose 2000 --model_name baseline_adv_combine_two


python preprocess.py --model_name "robust_combine_three" -D empirical --edge
python preprocess.py --model_name "robust_combine_three" -D uniform --edge
python preprocess.py --model_name "robust_combine_three" -D centered --edge
python preprocess.py --model_name "robust_combine_three" -D empirical
python preprocess.py --model_name "robust_combine_three" -D uniform
python preprocess.py --model_name "robust_combine_three" -D centered

python preprocess.py --model_name "robust_monotonic" -D empirical --edge
python preprocess.py --model_name "robust_monotonic" -D uniform --edge
python preprocess.py --model_name "robust_monotonic" -D centered --edge
python preprocess.py --model_name "robust_monotonic" -D empirical 
python preprocess.py --model_name "robust_monotonic" -D uniform 
python preprocess.py --model_name "robust_monotonic" -D centered 

python preprocess.py --model_name "monotonic" -D empirical --edge
python preprocess.py --model_name "monotonic" -D uniform --edge
python preprocess.py --model_name "monotonic" -D centered --edge
python preprocess.py --model_name "monotonic" -D empirical 
python preprocess.py --model_name "monotonic" -D uniform 
python preprocess.py --model_name "monotonic" -D centered 

python preprocess.py --model_name "train_adv_combine" -D empirical --edge
python preprocess.py --model_name "train_adv_combine" -D uniform --edge
python preprocess.py --model_name "train_adv_combine" -D centered --edge
python preprocess.py --model_name "train_adv_combine" -D empirical 
python preprocess.py --model_name "train_adv_combine" -D uniform 
python preprocess.py --model_name "train_adv_combine" -D centered 

