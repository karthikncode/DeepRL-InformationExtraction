python server.py \
       --port 5050 \
       --outFile outputs/baseline \
       --modelFile trained_model.p \
       --entity 4 \
       --aggregate always \
       --trainEntities cached_entities/train.5.p\
       --testEntities cached_entities/dev+test.5.p 
