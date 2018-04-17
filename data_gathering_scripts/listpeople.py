# encoding: utf-8

# Write down the list of people who say something in the hearing dataset

import os

input_file = "/data/WorkData/ideology_from_audio/RESULTS/wordssyllablesformants.txt"
output_file = "/data/WorkData/ideology_from_audio/RESULTS/listpeople.txt"

listofwords = open(input_file, "r")
peoples = open(output_file, "a")

lines = listofwords.readlines()

names = []

for line in lines:
    name = line.split(',')[1]
    if not (name in names):
        names.append(name)

names.sort()

for name in names:
    peoples.write(name+"\n")


listofwords.close()

peoples.close()
