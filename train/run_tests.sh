#!/bin/bash
 
#SBATCH --account=temfom0  # Specify the account to charge

#SBATCH --job-name=train_models_and_test_monotonicity  # Job name

#SBATCH --output=my_job_%j.out  # Standard output and error log

#SBATCH --error=my_job_%j.err  # Error file. '%j' is replaced with the job ID

#SBATCH --ntasks=1  # Run on a single task

#SBATCH --cpus-per-task=32  # Number of CPU cores per task

#SBATCH --time=23:00:00  # Time limit hrs:min:sec

#SBATCH --partition=standard  # Specify the partition to submit to

#train monotonic
python monotonic.py --num_trees 1000 --model_name model_1000learner
<<<<<<< HEAD
=======
#train robust combine two
>>>>>>> master
time python train_insert_monotonic.py --train robustness_spec/seed_train_malicious/mutate_insert_any_pt1/pickles --model_name robust_combine_two
#train robust combine three
time python train_combine_monotonic.py --train robustness_spec/seed_train_malicious/mutate_insert_any_pt1/pickles --model_name robust_combine_three
#train robust monotonic
time python train_insert_monotonic.py --model_name robust_monotonic
#train_adv_combine
time python train_adv_combine.py --batch_size 50 --batches 132900 --verbose 2000 --model_name baseline_adv_combine_two
<<<<<<< HEAD
#baseline
#python train.py --baseline --batches 5276 
=======
# train baseline
python train.py --baseline --batches 5276 
>>>>>>> master

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

python preprocess.py --model_name "robust_combine_two" -D empirical --edge
python preprocess.py --model_name "robust_combine_two" -D uniform --edge
python preprocess.py --model_name "robust_combine_two" -D centered --edge
python preprocess.py --model_name "robust_combine_two" -D empirical
python preprocess.py --model_name "robust_combine_two" -D uniform
python preprocess.py --model_name "robust_combine_two" -D centered

python preprocess.py --model_name "robust_combine_two" -D empirical --edge
python preprocess.py --model_name "robust_combine_two" -D uniform --edge
python preprocess.py --model_name "robust_combine_two" -D centered --edge
python preprocess.py --model_name "robust_combine_two" -D empirical
python preprocess.py --model_name "robust_combine_two" -D uniform
python preprocess.py --model_name "robust_combine_two" -D centered


#Rscript -e 'install.packages(c("data.table","huxtable"), repos="https://cloud.r-project.org",lib="~/R/x86_64-redhat-linux-gnu-library/3.6")'
Rscript  paper_plots.R
