import train
import predict
import evaluate
from multiprocessing import Pool
import os


def try_params(params):
	training_file = "../data/tagged_data/whole_text_full_city/train.tag"

	c = params[0]
	prev_n = params[1]
	next_n = params[2]
	prune = params[3]
	pid = os.getpid()
	testing_file = "../data/tagged_data/whole_text_full_city/dev.tag"
	trained_model = train.main(training_file,"",prev_n,next_n, c, prune)
	predict.main(trained_model,testing_file,False,"output{0}.tag".format(pid),"output{0}.pred".format(pid))
	result = evaluate.entity_level_eval("output{0}.tag".format(pid), testing_file)
	print params, result[2]
	return result

def find_max(results, possible_params):
	best_params = None
	max_params = 0
	for i in range(len(results)):
		f1 = sum(results[i][2])/4
		if f1 > max_params:
			max_params = f1
			best_params = possible_params[i]
	return best_params, max_params

if __name__ == '__main__':
	#parameters: C, previous_n, next_n, prune
	possible_c = [10,100,1000]
	possible_prev_n = [0,1,2,3,4]
	possible_next_n = [0,1,2,3,4]
	possible_prune = [0,1,2,3,4]

	possible_params = []
	for c in possible_c:
		for prev_n in possible_next_n:
			for next_n in possible_next_n:
				for prune in possible_prune:
					possible_params.append([c,prev_n,next_n,prune])

	#try_params(possible_params[6])
	p = Pool(3)
	results = p.map(try_params, possible_params)
	best_params, max_params = find_max(results, possible_params)
	print results
	print best_params, max_params
