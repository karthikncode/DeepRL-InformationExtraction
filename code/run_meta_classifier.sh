python server.py \
       --port 5050 \
       --outFile outputs/baseline \
       --modelFile trained_model2.p \
       --entity 4 \
       --aggregate always \
       --trainEntities consolidated/train+context.5.p\
       --testEntities consolidated/dev+test+context.5.p\
       --classifierEval True
