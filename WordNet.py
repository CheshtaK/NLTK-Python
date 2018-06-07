from nltk.corpus import wordnet

synonym = wordnet.synsets("program")

print(synonym[0].name())
print(synonym[0].lemmas()[0].name())
print(synonym[0].definition())
print(synonym[0].examples())


synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))


#Wu and Palmer Method
w1 = wordnet.synset('ship.n.01')

w2 = wordnet.synset('boat.n.01')
print(w1.wup_similarity(w2))

w2 = wordnet.synset('car.n.01')
print(w1.wup_similarity(w2))

w2 = wordnet.synset('cat.n.01')
print(w1.wup_similarity(w2))
