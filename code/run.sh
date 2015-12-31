# script to run experiments

i=0
MAX_PARALLEL=20
for trainFile in "train" "dev"; do
  for testFile in "dev" "test"; do   
    if [ $trainFile != $testFile ]; then      
      for aggregate in "always" "conf"; do             
        for delayedReward in "False" "True"; do                     
          for entity in 0 1 2 3 4; do
            # echo $i && sleep $i &
            port=$(( $i+5050 ));
            name=$port.$trainFile.$testFile.$entity.$aggregate.$delayedReward;
            python server.py --port $port --trainFile downloaded_articles/$trainFile.extra \
                             --testFile downloaded_articles/$testFile.extra \
                             --trainEntities cached_entities/$trainFile.entities \
                             --testEntities cached_entities/$testFile.entities \
                             --outFile outputs/$name.out \
                             --modelFile trained_model.p \
                             --entity $entity --aggregate $aggregate \
                             --delayedReward $delayedReward &
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
    fi;
  done;
done;

# python server.py --port 5050 --trainFile downloaded_articles/train.extra --outFile outputs/check_city --modelFile trained_model.p --entity 3 --aggregate always --trainEntities cached_entities/train.entities
