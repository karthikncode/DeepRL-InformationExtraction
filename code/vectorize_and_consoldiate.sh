
# for file in "dev" "test" "train"; do 

python vec_consolidate.py dloads/EMA2/train.extra 4 trained_model.EMA.p consolidated/vec_train+dev.EMA_k.p;  
python consolidate.py dloads/EMA2/train.extra,dloads/EMA2/dev.extra 4 trained_model.EMA.p  consolidated/train+dev+context.EMA_k.p consolidated/vec_train+dev.EMA_k.p &
python consolidate.py dloads/EMA2/test.extra 4 trained_model.EMA.p  consolidated/test+context.EMA_k.p consolidated/vec_train+dev.EMA_k.p &

