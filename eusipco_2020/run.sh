# python3 -m benchmark_verification.main_train_CNN --num-epochs 1
# python3 -m benchmark_verification.main_train_CNN --num-epochs 40
# python3 -m benchmark_verification.main_train_CNN --num-epochs 100 --batch-size 8 --type none
# python3 -m benchmark_verification.main_train_CNN --num-epochs 100 --batch-size 8 # loss aamp

# few-shot learning
# python3 -m benchmark_verification.few_shot_train --N-way 5 --K-shot 5

# 11K hands dataset
# python3 -m benchmark_verification.main_train_11K --num-epochs 100 --batch-size 8 --learning-rate 0.01 #--type none
python3 -m benchmark_verification.main_train_11K --num-epochs 100 --batch-size 4 --learning-rate 0.01 --type none

# python3 -m benchmark_verification.main_train_CNN --evaluate --outdir palm_model
