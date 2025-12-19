# python3 -m benchmark_verification.main_train_CNN --num-epochs 1
# python3 -m benchmark_verification.main_train_CNN --num-epochs 40
# python3 -m benchmark_verification.main_train_CNN --num-epochs 100 --batch-size 8 --type none
# python3 -m benchmark_verification.main_train_CNN --num-epochs 100 --batch-size 8 # loss aamp
# python3 -m benchmark_verification.main_train_CNN --evaluate --outdir palm_model

# few-shot learning
# python3 -m benchmark_verification.few_shot_train --N-way 5 --K-shot 5

#######################################################
# above is deprecated, please use the following scripts

# 11K hands dataset
# python3 -m benchmark_verification.main_train --train-dataset 11K --eval-dataset 11K --num-epochs 100 --num-classes 600 --batch-size 4 --learning-rate 0.001 --type none --outdir results/model_11K
# python3 -m benchmark_verification.main_train --train-dataset 11K --eval-dataset tongji --evaluate --outdir results/model_11K

# Tongji dataset
# python3 -m benchmark_verification.main_train --train-dataset tongji --eval-dataset tongji --num-epochs 100 --batch-size 4 --learning-rate 0.01 --type none --outdir results/model_tongji
# python3 -m benchmark_verification.main_train --train-dataset tongji --eval-dataset 11K --evaluate --outdir results/model_tongji

# different loss type
python3 -m benchmark_verification.main_train --train-dataset 11K --eval-dataset 11K --num-epochs 100 --num-classes 600 --batch-size 4 --learning-rate 0.001 --type norm --outdir results/model_11K_norm
python3 -m benchmark_verification.main_train --train-dataset 11K --eval-dataset tongji --evaluate --outdir results/model_11K_norm
# python3 -m benchmark_verification.main_train --train-dataset 11K --eval-dataset 11K --num-epochs 100 --num-classes 600 --batch-size 4 --learning-rate 0.001 --type aamp --outdir results/model_11K_aamp
# python3 -m benchmark_verification.main_train --train-dataset 11K --eval-dataset tongji --evaluate --outdir results/model_11K_aamp
