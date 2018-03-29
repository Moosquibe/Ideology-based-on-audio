####
Gathers the data
####

def treat_file(filetotreat):
    lines = filetotreat.readlines()

    i=3 # Skip the first three lines
    current_word = (line[3])[2]
    word_occurrence = []

    while(i < len(lines)):
        collapsed_line = collapse_line[lines[i]]
        if (collapsed_line[0] == current_word):
            word_occurrence.append(collapsed_line)
            current_word = current_word
        else:
            if len(word_occurrence > 1): ## We only keep words with more than one syllable
                add_occurrence(current_word, word_occurrence, speaker, output)

            current_word = collapsed_line[0]
            word_occurence = [collapsed_line]
            i+=1
    ### Last word of the file
    if len(word_occurrence > 1):
    add_occurrence(current_word, word_occurrence, speaker, output)


def add_occurrence(word, occurence, speaker, output):
        output.write(word, speaker, str(occurrence))

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

output_file = open("result.txt", "a")
filetotreat = open("2012_11_184_s01_norm.txt")
