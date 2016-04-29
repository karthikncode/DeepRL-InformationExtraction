# script to run experiments

i=0
MAX_PARALLEL=20 
START_PORT=8000
TAGS="ema1"
modelFile=trained_model.EMA.p
for trainFile in "train+context"; do
  for testFile in "dev+test+context"; do   
    if [ $trainFile != $testFile ]; then      
      for aggregate in "majority" "always" "conf"; do             
        for delayedReward in "False" "True"; do                     
          for contextType in 0 1 2; do 
            for entity in 3; do
              port=$(( $i+ $START_PORT));
              name=$port.$TAGS.$trainFile.$testFile.$entity.$aggregate.$delayedReward.$contextType;
              python server.py --port $port \
                               --trainEntities consolidated/$trainFile.EMA_k.p \
                               --testEntities consolidated/$testFile.EMA_k.p \
                               --outFile outputs/$name.out \
                               --modelFile $modelFile \
                               --entity $entity --aggregate $aggregate \
                               --shooterLenientEval True \
                               --delayedReward $delayedReward \
                               --contextType $contextType >analysis/$name.analysis &
              cd dqn
              ./run_cpu $port logs/$name/ &
              cd ..

              ((i++));
              if (($i % $MAX_PARALLEL == 0)); then
                  wait;
              fi;                    
            done;
          done;   
        done;
      done;
    fi;
  done;
done;

# python server.py --port 5050 --trainFile downloaded_articles/train.extra --outFile outputs/check_city --modelFile trained_model.p --entity 3 --aggregate always --trainEntities cached_entities/train.entities
