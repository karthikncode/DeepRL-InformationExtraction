import pickle
from collections import defaultdict

cities = defaultdict(set)
stop_words = set(['san', 'de', 'club', 'grand', 'lee', 'superior', 'safety', 'news', 'and', 'the', 'of', 'town', 'university', 'thomas', 'north', 'south', 'east', 'west', 'in', 'district', 'neighbors', 'new', 'john', 'lynn', 'upper', 'lower', 'king', 'van', 'court', 'park','city','lake','point','center','station','hill', 'la', 'valley', 'spring', 'beach'])

i = 0
for line in open('Top5000Population.csv'):
	if i > 3000:
		break
	city = line.split(",")[0]
	words = city.strip(' \t').split()
	if len(words) == 1:
		if words[0].lower() not in stop_words:
			cities[words[0]].add("")
	else:
		cities[words[0]].add(words[1])
	#if word.lower() not in stop_words:
	#	cities.add(word)
	i+=1

pickle.dump(cities, open("../data/constants/cities.p", "wb"))
