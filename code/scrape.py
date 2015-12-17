from newspaper import Article
import csv
import random
import os
import inflect
import re
import time

p = inflect.engine()
tags_text = ['killedNum','woundedNum','city']

#ratios: [train,dev,test]
def setup(metadata_path,new_metadata_path,out_path_raw,first_paragraph_only,extract_summary):
	file = open(metadata_path,'rb')
	data = csv.reader(file,delimiter=',')
	table = [row for row in data]

	f = open(new_metadata_path,'w')
	
	global valid_articles_metadata
	valid_articles_metadata = [] #list of article metadata, each article is ['url',[tags_vals]]
	
	count = 0
	for i in range(1,len(table)):
		row = table[i]
		if row[3] != row[4]:
			for j in range(6,len(row)):
				if (len(row[j]) != 0):
					count += 1
					status,title,text = download_article(row[j],first_paragraph_only,extract_summary)
					if status:
						valid_articles_metadata.append([row[j],i-1,len(valid_articles_metadata),row[2],row[3],row[4],row[5].split(",")[0],title])
						raw_writer = open(out_path_raw + str(len(valid_articles_metadata)-1) + ".txt","w")
						raw_writer.write(text)
						raw_writer.close()
	file.close()

	writer = csv.writer(f)
	writer.writerow(['url', 'event_id','article_id','shooter_name','killed_num','wounded_num','city','title'])
	map(writer.writerow,valid_articles_metadata)
	f.close()

	print "%d urls, %d valid" %(count, len(valid_articles_metadata))

# download article from url
# input: url(String)
# output; (status(Boolean), title(String), main_text(String))
def download_article(url,first_paragraph_only,extract_summary):
	article = Article(url,language='en')
	article.download()
	article.parse()
	title = article.title.encode("utf-8",'ignore')
	if first_paragraph_only:
		main_text = article.text.encode("utf-8",'ignore').split("\n")[0]
	elif extract_summary:
		try:
			article.nlp()
		except:
			return (False,"","")
		else:
			main_text = article.summary.encode("utf-8",'ignore').replace("\n", " ")
	else :
		main_text = article.text.encode("utf-8",'ignore').replace("\n", " ")

	if len(title.split(" ")) <=5 or len(main_text) <= 100:
		return (False,"","",None,"")
	else:
		return (True,title,main_text,article.publish_date,title)


def write_tags_to_file(t):
	ids = t[0]
	out_path = t[1]
	file_lines = map(gen_tags_file,ids)
	if len(file_lines) > 0:
		writer = open(out_path,"w")
		writer.write("\n".join(file_lines))
		writer.write("\n")
		writer.close()

def gen_all_tags(metadata_path,out_base_path,file_path,ratios):
	global file_base_path
	file_base_path = file_path
	file = open(metadata_path,'rb')
	data = csv.reader(file,delimiter = ",")

	global articles_metadata
	articles_metadata = [row for row in data]
	articles_metadata.pop(0)

	articles_num = len(articles_metadata)

	train_size = int(articles_num*ratios[0])
	dev_size = int(articles_num*ratios[1])
	train_set_idx = random.sample(xrange(articles_num),train_size)
	dev_set_idx = random.sample([x for x in xrange(articles_num) if x not in train_set_idx],dev_size)
	text_set_idx = [x for x in xrange(articles_num) if x not in train_set_idx + dev_set_idx]

	map(write_tags_to_file,[[train_set_idx,out_base_path+"train.tag"],[dev_set_idx,out_base_path+"dev.tag"],[text_set_idx,out_base_path+"test.tag"]])


def gen_tags_file(row_id):
	row = articles_metadata[row_id]
	file_id = row[2]
	f = open(file_base_path + file_id+'.txt','rb')
	text = f.read()
	f.close()
	title = row[7]
	text = title + ' ' + text
	text = text.replace("_"," ")
	words = re.findall(r"[\w']+|[.,!?;]", text)
	normalised_words = map(normalise_word, words)
	if row[3].lower() == "unknown":
		shooter_names = []
	else:
		shooter_names = row[3].replace(',','').replace('and', '').split(' ')
	for shooter_name in list(shooter_names):
		if shooter_name.lower() not in normalised_words:
			shooter_names.remove(shooter_name)
	tags_vals = [p.number_to_words(int(row[4])),p.number_to_words(int(row[5])),row[6].lower()]
	tags_vals_int = [int(row[4]),int(row[5])]
	correct_entities = row[4:]
	for i in [0,1]:
		entity = correct_entities[i]
		if entity.isdigit():
			entity = p.number_to_words(int(entity))
		#if entity != "zero" and entity not in normalised_words:
		#	correct_entities[i] = ""
	cities = correct_entities[2].split(" ")
	# for city_word in cities:
	# 	if city_word.lower() not in normalised_words:
	# 		correct_entities[2] = ""
	# 		break

	title_text = '|'.join(shooter_names) + "," + ",".join(correct_entities)
	return  title_text + '\n' + gen_tags(words,tags_vals,tags_vals_int,shooter_names)

def gen_tags(words,tags_vals,tags_vals_int,shooter_names):
	result = []
	for word in words:
		is_special_tag = False
		for i in range(2):
			if word.lower() == tags_vals[i] and word.lower() != "unknown":
				result.append(word + '_' + tags_text[i])
				is_special_tag = True
				break
		for j in range(2):
			if word == str(tags_vals_int[j]):
				result.append(word + '_' + tags_text[j])
				is_special_tag = True
				break
		if word.lower() in tags_vals[2].split(" ") and word.lower() != 'unknown' and word.lower() != 'undisclosed':
			result.append(word + '_city')
			is_special_tag = True
		if word in shooter_names:
			result.append(word + '_' + 'shooterName')
			is_special_tag = True
		if not is_special_tag:
			result.append(word + '_TAG')
	return " ".join(result)

def normalise_word(word):
	if word.isdigit():
		return p.number_to_words(int(word))
	return word.lower()

if __name__ == "__main__":
	# setup('../data/metadata/original/2013MASTER.csv','../data/metadata/text_summary/2013.csv',"../data/raw_data/text_summary/2013/",False,True)
	# setup('../data/metadata/original/2014MASTER.csv','../data/metadata/text_summary/2014.csv',"../data/raw_data/text_summary/2014/",False,True)
	# setup('../data/metadata/original/2015CURRENT.csv','../data/metadata/text_summary/2015.csv',"../data/raw_data/text_summary/2015/",False,True)
    gen_all_tags('../data/metadata/whole_text_full_city/2013.csv',"../data/tagged_data/whole_text_full_city/","../data/raw_data/whole_text_full_city/2013/",[1.0,0.0,0.0])
    gen_all_tags('../data/metadata/whole_text_full_city/2014.csv',"../data/tagged_data/whole_text_full_city/","../data/raw_data/whole_text_full_city/2014/",[1.0,0.0,0.0])
    gen_all_tags('../data/metadata/whole_text_full_city/2015.csv',"../data/tagged_data/whole_text_full_city/","../data/raw_data/whole_text_full_city/2015/",[0.0,0.5,0.5])






