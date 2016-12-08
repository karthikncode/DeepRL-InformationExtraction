## Information Extraction with Reinforcement Learning

### Installation
You will need to install [Torch](http://torch.ch/docs/getting-started.html) and the  python packages in `requirements.txt`.  

You will also need to install the Lua dev library `liblua` (`sudo apt-get install liblua5.2`) and the [signal](https://github.com/LuaDist/lua-signal) package for Torch to deal with SIGPIPE issues in Linux.
(You may need to uninstall the [signal-fft](https://github.com/soumith/torch-signal) package or rename it to avoid conflicts.)

### Data Preparation

Create the vectorizers (using a pre-trained model), for example:  
`python vec_consolidate.py dloads/Shooter/train.extra 5 trained_model2.p consolidated/vec_train.5.p`   
`python vec_consolidate.py dloads/Shooter/dev.extra 5 trained_model2.p consolidated/vec_dev.5.p`   
  
Consolidate the articles, for example:  
`python consolidate.py dloads/Shooter/train.extra 5 trained_model2.p consolidated/train+context.5.p consolidated/vec_train.5.p`  
`python consolidate.py dloads/Shooter/dev.extra 5 trained_model2.p consolidated/dev+context.5.p consolidated/vec_dev.5.p`  


### Running the code
  * Change to the code directory: `cd code/`
  * First run the server, for example:  
    `python server.py --port 7000 --trainEntities consolidated/train+context.5.p --testEntities consolidated/dev+context.5.p --outFile outputs/run.out --modelFile trained_model2.p --entity 4 --aggregate always --shooterLenientEval True --delayedReward False --contextType 2` 

  * In a separate terminal/tab, change to the agent code directory: `cd code/dqn/`
  * Then run the agent:  
    `./run_cpu 7000 logs/tmp/`  
    Make sure the port numbers for the server and agent match up.

### Acknowledgements
  * [Deepmind's DQN codebase](https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner)

