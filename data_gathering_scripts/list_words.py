# encoding: utf-8
import os

input_file = "/data/WorkData/ideology_from_audio/RESULTS/wordssyllablesformants.txt"
output_file = "/data/WorkData/ideology_from_audio/RESULTS/listwords.txt"

listofwords = open(input_file, "r")
orderedwords = open(output_file, "a")

lines = listofwords.readlines()

words = {}

for line in lines:
    word = line.split(',')[0]
    if (word in words):
        words[word] += 1
    else:
        words[word] = 1

listofwords.close()

words_rev = [(words[word], word) for word in words]
words_rev.sort(reverse=True)

for value, word in words_rev:
    if (value > 100):
        orderedwords.write(word+" "+str(value)+"\n")
    else:
        break
orderedwords.close()
