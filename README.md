# On Training Robust PDF Malware Classifiers

Code for our paper [On Training Robust PDF Malware Classifiers](https://arxiv.org/abs/1904.03542) (Usenix Security'20)
Yizheng Chen, Shiqi Wang, Dongdong She, Suman Jana

## Dataset

#### Full PDF dataset

Available [here at contagio](http://contagiodump.blogspot.com/2013/03/16800-clean-and-11960-malicious-files.html): "16,800 clean and 11,960 malicious files for signature testing and research."

#### Training and Testing datasets

We split the PDFs into 70% train and 30% test. Then, we used the [Hidost feature extractor](https://github.com/srndic/hidost) to
extract structural paths features, with the default `compact` path option.
We obtained the following training and testing data.

|   | Training PDFs  | Testing PDFs  |
|---|---|---|
| Malicious | 6,896 | 3,448 |
| Benign | 6,294 | 2,698 |

The hidost structural paths are [here](https://github.com/surrealyz/pdfclassifier/tree/master/data/extracted_structural_paths).

The extracted training and testing libsvm files are [here](https://github.com/surrealyz/pdfclassifier/tree/master/data).

[500 seed malware hash list.](https://github.com/surrealyz/pdfclassifier/blob/master/data/seeds_hash_list.txt)

## Models

The following models are TensorFlow checkpoints, except that two ensemble models need additional wrappers.

| Checkpoint |  Model |
|---|---|
| baseline_checkpoint  | Baseline  |
| baseline_adv_delete_one  | Adv Retrain A  |
| baseline_adv_insert_one  | Adv Retrain B  |
| baseline_adv_delete_two  | Adv Retrain C  |
| baseline_adv_insert_rootallbutone  | 	Adv Retrain D  |
| baseline_adv_combine_two  | Adv Retrain A+B  |
| adv_del_twocls  | Ensemble A+B Base Learner  |
| adv_keep_twocls  | Ensemble D Base Learner  |
| robust_delete_one  | Robust A  |
| robust_insert_one  | Robust B  |
| robust_delete_two  | Robust C  |
| robust_insert_allbutone  | Robust D  |
| robust_monotonic  | Robust E  |
| robust_combine_two_v2_e18  | Robust A+B  |
| robust_combine_three_e17  | Robust A+B+E  |

The following are XgBoost tree ensemble models.

| Binary  | Model  |
|---|---|
| model_10learner_test.bin  | Monotonic Classifier, 10 learners  |
| model_100learner.bin  | Monotonic Classifier, 100 learners  |
| model_1000learner.bin  | Monotonic Classifier, 1000 learners  |
| model_2000learner.bin  | 	Monotonic Classifier, 2000 learners  |

## Training Code

## Baseline Comparison

## Attacks in the Paper

## MalGAN Attack Evaluation

Please check out this [MalGAN attack evaluation](https://github.com/xiaoluLucy814/Malware-GAN-attack) against our robust models by [Zeyi](https://github.com/xiaoluLucy814/).
