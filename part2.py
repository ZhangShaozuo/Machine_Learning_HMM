from collections import defaultdict
import sys

# get emission parameter matrix, return a nested dict
def estimate_emission_parameters(input_file):
    state_value_pair = {}
    state_count = {}
    words = []
    with open(input_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            l = line.strip('\n').split(' ')
            if len(l)!=2:
                continue
            if l[1] not in state_value_pair:
                state_value_pair[l[1]] = {}
            if l[0] not in state_value_pair[l[1]]:
                state_value_pair[l[1]][l[0]] = 1
            else:
                state_value_pair[l[1]][l[0]] += 1
            if l[1] in state_count:
                state_count[l[1]] += 1
            else:
                state_count[l[1]] = 1
            if l[0] not in words:
                words.append(l[0])
    f.close()
    def getResult(state_value, state):
        res = {}
        keys = state.keys()
        for i in keys:
            res[i] = {}
        for key in state:
            for word in state_value[key]:
                if state[key]>0:
                    res[key][word] = state_value[key][word]/state[key]
        return res
    data = getResult(state_value_pair, state_count)
    for key in data:
        for word in words:
            if word not in data[key]:
                data[key][word] = 0
    return words, data

# get fixed emission parameter matrix, return a nested dict
def estimate_emission_parameters_fixed(input_file):
    state_value_pair = {}
    state_count = {}
    words = []
    with open(input_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            l = line.strip('\n').split(' ')
            if len(l)!=2:
                continue
            if l[1] not in state_value_pair:
                state_value_pair[l[1]] = {}
            if l[0] not in state_value_pair[l[1]]:
                state_value_pair[l[1]][l[0]] = 1
            else:
                state_value_pair[l[1]][l[0]] += 1
            if l[1] in state_count:
                state_count[l[1]] += 1
            else:
                state_count[l[1]] = 1
            if l[0] not in words:
                words.append(l[0])
    f.close()
    def getResult(state_value, state):
        res = {}
        keys = state.keys()
        for i in keys:
            res[i] = {}
        for key in state:
            for word in state_value[key]:
                if state[key]>0:
                    res[key][word] = state_value[key][word]/(state[key]+0.5)
            res[key]['#UNK#'] = 0.5/(state[key]+0.5) 
        return res
    data = getResult(state_value_pair, state_count)
    for key in data:
        for word in words:
            if word not in data[key]:
                data[key][word] = 0
    return words, data

# get the max probability of state of each word
def get_max_prob(input_file):
    words, metadata = estimate_emission_parameters_fixed(input_file)
    res = {}
    for word in words:
        l = [(key, metadata[key][word]) for key in metadata]
        res[word] = max(l, key=lambda i : i[1])[0]
    l = [(key, metadata[key]["#UNK#"]) for key in metadata]
    res["#UNK#"] = max(l, key=lambda i : i[1])[0]
    return res

# # MAIN FUNCTION
if len(sys.argv) < 4:
    print ('Please make sure you have installed Python 3.4 or above!')
    print ("python part2.py training_file input_file output_file")
    sys.exit()

training_file = sys.argv[1]
input_file = sys.argv[2]
output_file = sys.argv[3]

tags_produced = get_max_prob(training_file)
with open(input_file, 'r', encoding='UTF-8') as fin:
    with open(output_file, 'w', encoding='UTF-8') as fout:
        for line in fin.readlines():
            line = line.strip()
            if len(line)==0:
                fout.write('\n')
            else:
                if line in tags_produced:
                    fout.write(line+" "+tags_produced[line]+"\n")
                else:
                    fout.write(line+" "+tags_produced["#UNK#"]+"\n")
    fout.close()
fin.close()
print("Execution Finished")