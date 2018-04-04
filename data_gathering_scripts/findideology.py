# encoding: utf-8
file_people = open("listpeople.txt", 'r')
list_people = file_people.readlines()

file_donations = open("contribDB_2008.csv", 'r')

log = open('log.txt', 'a')
speakers = []

for people in list_people:
    name_components = people.split()
    if len(name_components) == 2:
        first_name = name_components[0]
        last_name = name_components[1]
        middle_name = "EMPTY"
    elif len(name_components) > 2:
        first_name = name_components[0]
        middle_name = name_components[1]
        last_name = name_components[2]
    speakers.append({'FN':first_name, 'LN':last_name, 'ideology':[]})
file_people.close()

for i in range(25000000):
    donation = file_donations.readline()
    if "judge" in donation or "lawyer" in donation or "attorney" in donation or "advocate" in donation or "law" in donation:
            name = donation.split(',"')[4]
            name_components = name.split(',')
            if (len(name_components) == 2):
                last_name = name_components[0].replace('"', '')
                first_names = (name_components[1].split())
                first_name = first_names[0].replace('"', '')
                party = donation.split(',')[-22]
                if party== '"100"':
                    ideo = "DEM"
                elif party== '"200"':
                    ideo = "REP"
                elif party == '"328"':
                    ideo = "IND"
                else:
                    ideo = "UNDEF"
                for speaker in speakers:
                    if speaker['FN'] == first_name and speaker['LN'] == last_name:
                        speaker['ideology'].append(ideo)

for speaker in speakers:
    name = speaker['FN']+" "+speaker['LN']
    if len(speaker['ideology']) == 0:
        log.write("For "+name+" we have no idea \n")
    elif len(speaker['ideology']) == 1:
        log.write("For "+name+" it is clearly "+ideo+"\n")
    else:
        log.write("For "+name+" it is unclear \n")

file_donations.close()
log.close()
