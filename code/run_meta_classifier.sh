python server.py \
       --port 5050 \
       --outFile outputs/baseline \
       --modelFile trained_model2.p \
       --entity 4 \
       --aggregate always \
       --trainEntities consolidated/train+context.EMA.p\
       --testEntities consolidated/dev+context.EMA.p\
       --classifierEval True
