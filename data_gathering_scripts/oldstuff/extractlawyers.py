# encoding: utf-8
# Extract the lawyers from the donation dataset, to avoid looping on a huuuge file
import csv

DONATIONS_CSV = "/Users/thomasl/Documents/GitHub/contribDB_president.csv"
DONATIONS_LAWYERS = "contriblawyers_from_presidential.csv"

file_donations=open(DONATIONS_CSV, 'r')
file_contributions=open(DONATIONS_LAWYERS, 'w')

def is_lawyer(job):
    return ("judge" in job or "lawyer" in job or "attorney" in job or "advocate" in job or "law" in job or "justice" in job)


contributions = csv.writer(file_contributions)

for donation in csv.reader(file_donations):
    if is_lawyer(donation[19]):
        contributions.writerow(donation)

file_donations.close()
file_contributions.close()
