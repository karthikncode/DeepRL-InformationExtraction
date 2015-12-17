from __future__ import division
import sys
import numpy as np
from collections import defaultdict
import inflect
from train import load_data
import warnings

inflect_engine = inflect.engine()

def entity_level_eval(output_file, correct_file):
	'''
	output_file := output file from training
	'''
	
	TP, FP, FN = np.zeros(5), np.zeros(5), np.zeros(5)

	tags2int = {"TAG": 0, "shooterName":1, "killedNum":2, "woundedNum":3, "city":4}

	line_num = 1
	with open(correct_file) as cfile:
		with open(output_file) as ofile:
			for line1, line2 in zip(cfile, ofile):
				if line_num % 2 == 0:
					lines_correct = line1.split()
					lines_output = line2.split()
					for pair1, pair2 in zip(lines_correct, lines_output):
						word1 = pair1.split('_')[0]
						tag_correct = pair1.split('_')[1]
						word2 = pair2.split('_')[0]
						tag_output = pair2.split('_')[1]
						if word1 == word2: # this must be true each time
							if tag_correct == tag_output: # T.P. counted only if the tag is same, not just positive.
								TP[tags2int[tag_correct]] += 1
							else:
								FP[tags2int[tag_output]] += 1
								FN[tags2int[tag_correct]] += 1
						else:
							raise Exception("Erorr: file mismatch")
				line_num += 1

	TP = TP[1:]
	FP = FP[1:]
	FN = FN[1:]
	pre = TP/(TP+FP)
	re = TP/(TP+FN)
	F1 = 2*(pre*re/(pre+re))

	print "Precision: ", pre
	print "Recall: ", re
	print "F1: ", F1
	return [pre, re, F1]


def article_level_eval(output_file, test_tags):
	'''
	output_file := output file from training
	'''
	TP_shooterName, FP_shooterName, FN_shooterName = 0,0,0
	tags = ['shooterName', 'killedNum', 'woundedNum', 'city']
	other_tags = ['killedNum','woundedNum','city']
	other_tags_true = defaultdict(int)
	other_tags_total = defaultdict(int)
	other_tags_samples = defaultdict(int)
	shooterName_app = 0

	texts, identifiers = load_data(test_tags)

	line_num = 1
	with open(output_file,'r') as ofile:
		curr_correct_entities = {}
		predicted_entities = {}
		for line in ofile:
			if line_num % 2 == 1:
				curr_correct_entities = make_result_dict(line)
				# Now the following code is taken care of in scrape :)
				# original_text = texts[int((line_num-1)/2)][0]
				# original_text = map(str.lower,original_text)
				# original_text = map(convert_num_to_word,original_text)
				# for tag in curr_correct_entities.keys():
				# 	if tag in other_tags:
				# 		if tag == "city":
				# 			for entity in curr_correct_entities[tag].split(" "):
				# 				if entity != "" and entity not in original_text:
				# 					print "city", entity, line
				# 					curr_correct_entities[tag] = ""
				# 					other_tags_removed[tag] += 1

				# 		else:
				# 			entity = curr_correct_entities[tag]
				# 			if entity != "zero" and entity != "" and entity not in original_text:
				# 				print "number", entity, line
				# 				curr_correct_entities[tag] = ""
				# 				other_tags_removed[tag] += 1
				# 	else:
				# 		for entity in list(curr_correct_entities[tag]):
				# 			if entity not in original_text:
				# 				print "shooterName", entity, line
				# 				curr_correct_entities[tag].remove(entity)
				# 		if curr_correct_entities[tag] == []:
				# 			print tag
				# 			del curr_correct_entities[tag]
			else:
				predicted_entities = make_result_dict(line)
				for tag in other_tags:
					if curr_correct_entities[tag] != "" and curr_correct_entities[tag] != "zero":
						if len(predicted_entities[tag]) > 0:
							if curr_correct_entities[tag] == predicted_entities[tag]:
								other_tags_true[tag] += 1
							other_tags_total[tag] += 1
						other_tags_samples[tag] += 1

				if "shooterName" in curr_correct_entities:
					print curr_correct_entities['shooterName']
					shooterName_app += 1
					curr_TP = 0.0
					curr_FP = 0.0
					curr_FN = 0.0
					if "shooterName" in predicted_entities:
						for word in set(predicted_entities["shooterName"]):
							if word in curr_correct_entities["shooterName"]:
								curr_TP += 1
							else:
								curr_FP += 1
					for word in curr_correct_entities["shooterName"]:
						if "shooterName" not in predicted_entities or word not in predicted_entities["shooterName"]:
							curr_FN += 1
					if curr_TP > 0:
						TP_shooterName += curr_TP/len(predicted_entities["shooterName"])
					if curr_FP > 0:
						FP_shooterName += curr_FP/len(predicted_entities["shooterName"])
					if curr_FN > 0:
						FN_shooterName += curr_FN/len(curr_correct_entities["shooterName"])

					print curr_TP,curr_FP,curr_FN

			line_num += 1

	total_samples = (line_num-1) / 2

	print TP_shooterName,FP_shooterName,FN_shooterName
	if TP_shooterName != 0:
		shooter_pre = TP_shooterName/(TP_shooterName+FP_shooterName)
		shooter_re = TP_shooterName/(TP_shooterName+FN_shooterName)
		shooter_F1 = 2*(shooter_pre*shooter_re/(shooter_pre+shooter_re))
	else:
		shooter_pre = 0
		shooter_re = 0
		shooter_F1 = 0
	
	other_tags_pre = {} 
	other_tags_re = {}
	other_tags_prediction_rate = {}
	other_tags_f1 = {}

	print other_tags_total
	print total_samples

	for tag in other_tags:
		other_tags_pre[tag] = other_tags_true[tag] / other_tags_total[tag]
		other_tags_re[tag] = other_tags_true[tag]/ other_tags_samples[tag]
		other_tags_prediction_rate[tag] = other_tags_total[tag] / other_tags_samples[tag]
		other_tags_f1[tag]  = 2 * other_tags_pre[tag] * other_tags_re[tag] /(other_tags_pre[tag] + other_tags_re[tag])

	print "shooter appearance times", shooterName_app
	print "shooter appearance rate", shooterName_app /total_samples
	print "shooter precision: ", shooter_pre
	print "shooter recall: ", shooter_re
	print "shooter f1 score: ", shooter_F1
	print "precisions: ", other_tags_pre
	print "recalls: " , other_tags_re
	print "prediction_rate: ", other_tags_prediction_rate
	print "f1 scores: ", other_tags_f1

def make_result_dict(line):
	d = {}
	tags = line.rstrip('\n').split(',')
	if not (tags[0].lower() == "unknown" or tags[0].lower() == 'undisclosed'or len(tags[0]) == 0):
		d["shooterName"] = map(str.lower,tags[0].split('|'))
	d["killedNum"] = convert_num_to_word(tags[1])
	d["woundedNum"] = convert_num_to_word(tags[2])
	d["city"] = tags[3].lower()
	return d

def convert_num_to_word(s):
	if not s.isdigit():
		return s
	else:
		return inflect_engine.number_to_words(int(s))

if __name__ == '__main__':
	output_file = sys.argv[1]
	test_tags = "../data/tagged_data/whole_text_full_city/dev.tag"
	#entity_level_eval(output_file, test_tags)
	article_level_eval(output_file, test_tags)