from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

words = ["python","pythoner","pythoning","pythoned","pythonly"]

for w in words:
    print(ps.stem(w))

sentence = "It is important to by very pythonly while you are pythoning with python.All pythoners have pythoned poorly at least once."

sentence_words = word_tokenize(sentence)

for w in sentence_words:
    print(ps.stem(w))
    
