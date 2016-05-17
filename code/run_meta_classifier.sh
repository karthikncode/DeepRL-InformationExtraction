python server.py \
       --port 5050 \
       --outFile outputs/baseline \
       --modelFile trained_model.EMA.p \
       --entity 3 \
       --aggregate always \
       --trainEntities consolidated/train+dev+context.EMA_k.p\
       --testEntities consolidated/test+context.EMA_k.p\
       --classifierEval True
