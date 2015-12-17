import pickle
import inflect
p = inflect.engine()
words = set(['first','second','third','fourth','fifth','sixth','seventh','eighth','ninth','tenth','eleventh','twelfth','thirteenth','fourteenth','fifteenth',
	'sixteenth','seventeenth','eighteenth','nineteenth','twentieth','twenty-first','twenty-second','twenty-third','twenty-fourth','twenty-fifth'])

pickle.dump(words, open("../data/constants/word_ordinals.p", "wb"))