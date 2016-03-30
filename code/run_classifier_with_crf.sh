python server.py \
       --port 5050 \
       --outFile outputs/baseline \
       --modelFile trained_model2.p \
       --entity 4 \
       --aggregate always \
       --trainEntities consolidated/train+context.crf.p\
       --testEntities consolidated/dev+test+context.crf.p\
       --classifierEval True
