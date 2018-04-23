# encoding: utf-8

# Find ideology from a donation dataset

import csv

DONATIONS_CSV = "contriblawyers_from_presidential.csv"
PEOPLE_FILE = "listpeople.txt"
LOG_FILE = "results_ideologt_from_presidential.txt"


log = open(LOG_FILE, "w")

#### Form the list of speakers
file_people = open(PEOPLE_FILE, 'r')
list_people = file_people.readlines()

speakers = []

for people in list_people:
    name_components = people.split()
    if len(name_components) == 2:
        first_name = name_components[0].lower()
        last_name = name_components[1].lower()
    elif len(name_components) > 2:
        first_name = name_components[0].lower()
        last_name = name_components[2].lower()
    speakers.append({'FN':first_name, 'LN':last_name})

file_people.close()

print("Liste etablie")

def is_lawyer(job):
    return ("judge" in job or "lawyer" in job or "attorney" in job or "advocate" in job or "law" in job or "justice" in job)

def parcours_donation(speaker):
    sum_dem = 0
    sum_rep = 0
    file_donations = open(DONATIONS_CSV, 'r')
    for donation in csv.reader(file_donations):
        amount = float(donation[3])
        first_name = donation[8]
        last_name = donation[7]
        party = donation[24]
        job = donation[19]
        if first_name == speaker['FN'] and last_name == speaker['LN'] and is_lawyer(job):
            print("FOUND ONE!")
            print("An amount of "+str(amount))
            if party=="100":
                sum_dem += amount
                print("A democrat!")
            elif party=="200":
                sum_rep += amount
                print("A republican !")
    file_donations.close()
    print("Parcours termine !")
    if (sum_dem+sum_rep == 0):
        return -1
    else:
        return (sum_rep)/(sum_dem+sum_rep)


for speaker in speakers:
    ideology = parcours_donation(speaker)
    if ideology == -1:
        log.write(speaker['FN']+' '+speaker['LN']+' is undefined \n')
    else:
        log.write(speaker['FN']+' '+speaker['LN']+' is '+str(ideology)+' \n')

log.close()
