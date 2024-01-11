#!/bin/bash
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

