python server.py \
       --port 5050 \
       --outFile outputs/baseline \
       --modelFile trained_model.EMA.p \
       --entity 3 \
       --aggregate always \
       --trainEntities consolidated/train+context.EMA_crf.p\
       --testEntities consolidated/test+context.EMA_crf.p\
       --classifierEval True
