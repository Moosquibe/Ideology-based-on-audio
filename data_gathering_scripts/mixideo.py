 # encoding: utf-8
file_1 = "ideology.txt"
file_2 = "results_ideology_from_presidential.txt"
PEOPLE_FILE = "listpeople.txt"

f1 = open(file_1, "r")
f2 = open(file_2, "r")
f3 = open("ideologyfinal.txt", "w")


def addi(a,b):
    if a == "undefined" and b == "undefined":
        return "undefined"
    elif a == "undefined":
        return b
    elif b == "undefined":
        return a
    else:
        return (float(a)+float(b))/2

for line1 in f1.readlines():
    line2 = f2.readline()
    comp1 = line1.split()
    comp2 = line2.split()
    fn = comp1[0]
    ln = comp1[1]
    r1 = comp1[3]
    r2 = comp2[3]
    f3.write(fn+','+ln+','+str(addi(r1, r2))+'\n')


f1.close()
f2.close()
f3.close()
