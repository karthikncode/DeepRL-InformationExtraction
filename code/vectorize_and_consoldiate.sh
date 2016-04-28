
for file in "dev" "test" "train"; do 

    python vec_consolidate.py dloads/EMA/$file.extra 3 trained_model.EMA.p consolidated/vec_$file.EMA.p && 
    python consolidate.py dloads/EMA/$file.extra 3 trained_model.EMA.p  consolidated/$file+context.EMA.p consolidated/vec_$file.EMA.p &

done;


wait
echo "Consolidating complete"


# for file in "dev"; do 

#     python consolidate.py dloads/EMA/$file.extra 1 trained_model.EMA.p  consolidated/$file+context.EMA.p consolidated/vec_$file.EMA.p 
# done;


# wait
# echo "Consolidating complete"
