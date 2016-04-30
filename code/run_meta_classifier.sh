python server.py \
       --port 5050 \
       --outFile outputs/baseline \
       --modelFile trained_model.EMA.p \
       --entity 3 \
       --aggregate always \
       --trainEntities consolidated/train+context.EMA_k.p\
       --testEntities consolidated/dev+test+context.EMA_k.p\
       --classifierEval True
