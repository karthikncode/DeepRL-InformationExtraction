echo "Training Max ent classifier and reporting tag level scores";
python train.py;
echo "Collapsing neighboring tags and reporting entity level scores";
python predict.py;