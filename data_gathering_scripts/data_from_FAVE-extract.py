####
## Gathers the data from formants, using the names in the TextGrids
####
# encoding: utf-8
import os
import io

# TEXTGRIDS_FOLDERS = "/Users/thomasl/Documents/MLProject/FAVE-EXTRACT/"
TEXTGRIDS_FOLDERS = "/data/Dropbox/Data/Supreme_Court_Audio/Oyez_vowels/FAVE/oyez_full/"
FORMANTS_FOLDERS = "/data/Dropbox/Data/Supreme_Court_Audio/Oyez_vowels/FAVE/FAVE-extract/"
RESULT_FOLDERS = "/home/tleble/RESULTS/"


#######
### Takes a file with formants (the norm.txt ones)
### extracts the words with more than 1 syllable
### and adds (in 'ouput') a data entry of the type
### word, speaker, [list of the syllables and the formants data for each syllable]
#####
def treat_file(filetotreat, output, speaker):
    lines = filetotreat.readlines()
    current_word = "--"
    word_occurrence = []

    for line in (l.split() for l in lines[3:] if len(l.split()) == 26): ## Skip the first 3 lines, and avoid weird entries
        collapsed_line = collapse_line(line)
        if (collapsed_line[0] == current_word or current_word == "--"):
            word_occurrence.append(collapsed_line)
        else:
            if (len(word_occurrence) > 1): ## We only keep words with more than one syllable
                add_occurrence(current_word, word_occurrence, speaker, output)
            word_occurrence = [collapsed_line]
        current_word = collapsed_line[0]

    ### Last word of the file
    if (len(word_occurrence) > 1):
        add_occurrence(current_word, word_occurrence, speaker, output)

    return True

def add_occurrence(word, occurrence, speaker, output):
    output.write(word+','+speaker+','+str(occurrence)+'\n')

def collapse_line(line):
    vowel = line[0]
    # stress = line[1]
    word = line[2]
    formant_first = line[3]
    formant_second = line[4]
    # t 5, beg 6, end 7
    duration = line[8]
    # cd 9, fm 10, fp 11, fv 12, ps 13, fs 14, style 15, glide 16
    formant_first_20 = line[17]
    formant_second_20 = line[18]
    formant_first_35 = line[19]
    formant_second_35 = line[20]
    formant_first_50 = line[21]
    formant_second_50 = line[22]
    formant_first_65 = line[23]
    formant_second_65 = line[24]
    ff_summary = [formant_first_20, formant_first_35, formant_first_50, formant_first_65]
    fs_summary = [formant_second_20, formant_second_35, formant_second_50, formant_second_65]
    line_collapsed = [word, vowel, formant_first, formant_second, duration, ff_summary, fs_summary]
    return line_collapsed



if __name__ == "__main__":
    os.chdir(RESULT_FOLDERS)
    output = open("wordssyllablesformants.txt", "a")
    os.chdir(TEXTGRIDS_FOLDERS)
    years = os.listdir() ### List all the years
    for year in (y for y in years if y.isnumeric()):
        os.chdir(TEXTGRIDS_FOLDERS+year)
        list_hearings = os.listdir() ### List all the hearings for a given year
        for hearing in (h for h in list_hearings if h.endswith('.TextGrid')):
            hearing_textgrid = open(TEXTGRIDS_FOLDERS+year+'/'+hearing, 'r', encoding='utf-8')
            hearing_textgrid_lines = hearing_textgrid.readlines()
            item = 0
            for line in (l for l in hearing_textgrid_lines if l.strip().startswith("name =")): ### Looking for the speaker name
                item += 1
                speaker = (line.strip())[8:-1] ## Only keep the name
                if (item < 10):
                    name_file = hearing.strip('.TextGrid')+'_s0'+str(item)+'_norm.txt'
                else:
                    name_file= hearing.strip('.TextGrid')+'_s'+str(item)+'_norm.txt'
                try:
                    os.chdir(FORMANTS_FOLDERS+year+'_vowels') ### Opening the formant file
                    filetotreat = open(name_file)
                except FileNotFoundError:
                    break
                # os.chdir(RESULT_FOLDERS+year)
                # output = open("result_"+name_file, 'a')
                treat_file(filetotreat, output, speaker) #### Treating the formant file
                filetotreat.close()
    output.close()
