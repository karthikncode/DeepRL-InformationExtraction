# script to run experiments

i=0
MAX_PARALLEL=50
START_PORT=8000
TAGS="ema_params"
modelFile=trained_model.EMA.p
for trainFile in "train+context"; do
  for testFile in "dev+test+context"; do   
    if [ $trainFile != $testFile ]; then      
      for aggregate in "always"; do             
        for delayedReward in "False"; do                     
          for contextType in 0 1 2; do 
            for entity in 4; do
              for discount in 0.8 0.99; do 
                for learn_start in 50000; do 
                #for learn_start in 20000 50000; do 
                  for lr in 0.00005 0.000025 0.0001; do
                    for target_q in 5000 10000; do 
            # echo $i && sleep $i &
                    port=$(( $i+ $START_PORT));
                    name=$port.$TAGS.$trainFile.$testFile.$entity.$aggregate.$delayedReward.$contextType.$discount.$learn_start.$lr.$target_q;
                    python server.py --port $port \
                                     --trainEntities consolidated/$trainFile.EMA_k.p \
                                     --testEntities consolidated/$testFile.EMA_k.p \
                                     --outFile outputs/$name.out \
                                     --modelFile $modelFile \
                                     --entity $entity --aggregate $aggregate \
                                     --shooterLenientEval True \
                                     --delayedReward $delayedReward \
                                     --contextType $contextType &
                    cd dqn
                    ./run_cpu $port logs/$name/ $discount $learn_start $lr $target_q &
                    cd ..

                    ((i++));
                    if (($i % $MAX_PARALLEL == 0)); then
                        wait;
                    fi;                    
                    done;
                  done;
                done;
              done;
            done;
          done;   
        done;
      done;
    fi;
  done;
done;

# python server.py --port 5050 --trainFile downloaded_articles/train.extra --outFile outputs/check_city --modelFile trained_model.p --entity 3 --aggregate always --trainEntities cached_entities/train.entities
