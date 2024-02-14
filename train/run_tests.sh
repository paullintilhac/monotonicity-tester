#!/bin/bash

#to run this file in background and pipe output, run ./run_tests.sh > run_log.txt 2>&1 &
 
#SBATCH --account=temfom0  # Specify the account to charge

#SBATCH --job-name=train_models_and_test_monotonicity  # Job name

#SBATCH --output=my_job_%j.out  # Standard output and error log

#SBATCH --error=my_job_%j.err  # Error file. '%j' is replaced with the job ID

#SBATCH --ntasks=1  # Run on a single task

#SBATCH --cpus-per-task=32  # Number of CPU cores per task

#SBATCH --time=23:00:00  # Time limit hrs:min:sec

#SBATCH --partition=standard  # Specify the partition to submit to
: <<'END'
echo "Training Monotonic Model..."
#train monotonic
python monotonic.py --num_trees 1000 --model_name model_1000learner

echo "Training Robust (A+B) Model..."
#train robust combine two
time python train_insert_monotonic.py --train robustness_spec/seed_train_malicious/mutate_insert_any_pt1/pickles --model_name robust_combine_two

echo "Training Robust (A+B+E) Model..."
#train robust combine three
time python train_combine_monotonic.py --train robustness_spec/seed_train_malicious/mutate_insert_any_pt1/pickles --model_name robust_combine_three

echo "Training Robust (D) Model..."
#train robust monotonic
time python train_insert_monotonic.py --model_name robust_monotonic 

echo "Training Adversarially Trained (A+B) Model..."
#train_adv_combine
time python train_adv_combine.py  --batches 132900 --verbose 2000 --model_name baseline_adv_combine_two

echo "Training Baseline NN Model..."
# train baseline
python train.py --baseline --batches 5276

END

for n in 1 2 3 4 5 6 7 8 9 10;
do
python preprocess.py --model_name "robust_combine_three" -D empirical --edge --output_folder $n
python preprocess.py --model_name "robust_combine_three" -D centered --edge --output_folder $n
python preprocess.py --model_name "robust_combine_three" -D uniform --edge --output_folder $n
python preprocess.py --model_name "robust_combine_three" -D uniform --output_folder $n

python preprocess.py --model_name "robust_monotonic" -D empirical --edge --output_folder $n
python preprocess.py --model_name "robust_monotonic" -D uniform --edge --output_folder $n
python preprocess.py --model_name "robust_monotonic" -D centered --edge --output_folder $n
python preprocess.py --model_name "robust_monotonic" -D uniform  --output_folder $n

python preprocess.py --model_name "monotonic" -D empirical --edge --output_folder $n
python preprocess.py --model_name "monotonic" -D uniform --edge --output_folder $n
python preprocess.py --model_name "monotonic" -D centered --edge --output_folder $n
python preprocess.py --model_name "monotonic" -D uniform --output_folder $n

python preprocess.py --model_name "train_adv_combine" -D empirical --edge --output_folder $n
python preprocess.py --model_name "train_adv_combine" -D uniform --edge --output_folder $n
python preprocess.py --model_name "train_adv_combine" -D centered --edge --output_folder $n
python preprocess.py --model_name "train_adv_combine" -D uniform 

python preprocess.py --model_name "robust_combine_two" -D empirical --edge --output_folder $n
python preprocess.py --model_name "robust_combine_two" -D uniform --edge --output_folder $n
python preprocess.py --model_name "robust_combine_two" -D centered --edge --output_folder $n
python preprocess.py --model_name "robust_combine_two" -D uniform --output_folder $n

python preprocess.py --model_name "baseline" -D empirical --edge --output_folder $n
python preprocess.py --model_name "baseline" -D uniform --edge --output_folder $n
python preprocess.py --model_name "baseline" -D centered --edge --output_folder $n
python preprocess.py --model_name "baseline" -D uniform --output_folder $n

python preprocess.py --model_name "robust_combine_three" -D empirical --edge --train_only --output_folder $n
python preprocess.py --model_name "robust_combine_three" -D centered --edge --train_only --output_folder $n

python preprocess.py --model_name "robust_monotonic" -D empirical --edge --train_only --output_folder $n
python preprocess.py --model_name "robust_monotonic" -D centered --edge --train_only --output_folder $n

python preprocess.py --model_name "monotonic" -D empirical --edge --train_only --output_folder $n
python preprocess.py --model_name "monotonic" -D centered --edge --train_only --output_folder $n

python preprocess.py --model_name "train_adv_combine" -D empirical --edge --train_only --output_folder $n
python preprocess.py --model_name "train_adv_combine" -D centered --edge --train_only --output_folder $n

python preprocess.py --model_name "robust_combine_two" -D empirical --edge --train_only --output_folder $n
python preprocess.py --model_name "robust_combine_two" -D centered --edge --train_only --output_folder $n

python preprocess.py --model_name "baseline" -D empirical --edge --train_only --output_folder $n
python preprocess.py --model_name "baseline" -D centered --edge --train_only --output_folder $n
done
#Rscript -e 'install.packages(c("data.table","huxtable"), repos="https://cloud.r-project.org",lib="~/R/x86_64-redhat-linux-gnu-library/3.6")'
#Rscript  paper_plots.R

