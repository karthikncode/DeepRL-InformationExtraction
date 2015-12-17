from query import get_predictions_from_query
from predict import get_mode
from train import load_data
from collections import defaultdict
import helper

tags2int = {"TAG": 0, "shooterName":1, "killedNum":2, "woundedNum":3, "city":4}
int2tags = ["TAG",'shooterName','killedNum','woundedNum','city']
tags = [0,1,2,3,4]


def combine_predictions_batch(output_tags, output_predictions, output_combined_predictions):
    predictions = [line.rstrip('\n') for line in open(output_predictions)][1::2]
    texts, identifiers = load_data(output_tags)
    #identifiers = [line.rstrip('\n') for line in open(output_predictions)][::2]
    f = open(output_combined_predictions,'w')
    for i in range(len(predictions)):
        prediction = predictions[i]
        identifier = identifiers[i]
        text = texts[i]
        title = ",".join(identifier.split(",")[len(tags)-1:])
        #query_text = prediction.split(",")[tags2int["city"]-1] + " " + title
        query_text = title
        print "Query text", query_text
        #print text
        pred_list = get_predictions_from_query(query_text, " ".join(text[0]), "bing")
        pred_list.append(prediction)
        #pred_list.append(prediction)
        f.write(identifier)
        f.write("\n")
        f.write(combine_predictions(pred_list))
        print pred_list
        print combine_predictions(pred_list)
        f.write("\n")
    return

# Combines a list of prediction strings by taking the mode
def combine_predictions(pred_list):
    combined = defaultdict(list)
    for pred in pred_list:
        preds = pred.split(",")
        #print preds
        for i in range(1,len(int2tags)):
            tag = int2tags[i]
            if tag == "shooterName":
                combined[tag] += preds[i-1].split("|")
            else:
            	if preds[i-1] != "":
                	combined[tag].append(preds[i-1])
    output_pred_line = ""
    shooterNames = defaultdict(int)
    for shooterName in combined["shooterName"]:
    	if shooterName != "":
    		shooterNames[shooterName] += 1
    for shooterName in shooterNames:
        if shooterNames[shooterName] >= 2:
            output_pred_line += shooterName.lower()
            output_pred_line += "|"
    output_pred_line = output_pred_line[:-1]
    for tag in int2tags:
        if tag in ["killedNum", "woundedNum"]:
            output_pred_line += ","
            mode = get_mode(combined[tag])
            if mode == "":
                output_pred_line += "zero"
            else:
                output_pred_line += mode
        elif tag == "city":
            output_pred_line += ","
            output_pred_line += get_mode(combined[tag])
    return output_pred_line

if __name__ == "__main__":
    output_tags = "output.tag"
    output_predictions = "output.pred"
    output_combined_predictions = "output1.pred"
    combine_predictions_batch(output_tags, output_predictions, output_combined_predictions)