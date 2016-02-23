Information-Extraction
======================

Tag Names: TAG, shooterName, killedNum, woundedNum, city

======================
To run the pragram, cd into /code directory.

Samples:  
Create the vectorizers:  
`python vec_consolidate.py dloads/train.extra 5 trained_model2.p consolidated/vec_train.5.p`   
  
Consolidate the articles:  
`python consolidate.py dloads/train.extra 5 trained_model2.p consolidated/train+context.5.p consolidated/vec_train.5.p`  


Install
==========
https://github.com/LuaDist/lua-signal
