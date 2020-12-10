from part_4_buffer import Buffer
import sys
from tqdm import tqdm

# states for CN
# states = [
#     "NULL", "B-negative", "O", "B-neutral", "B-positive", "I-negative",
#     "I-neutral", "I-positive", "I-NP", "I-VP"
# ]
# states for EN(2)
states = [
    "NULL", "O", "I-NP", "I-VP", "I-ADVP", "I-ADJP", "B-NP", "B-VP", "B-ADVP",
    "B-ADJP", "I-PP", "B-PP", "I-SBAR", "B-SBAR", "I-PRT", "I-CONJP",
    "B-CONJP", "B-PRT", "I-INTJ", "B-INTJ", "I-UCP", "B-UCP", "I-LST", "B-LST"
]

iteration = 80
TOP_train = 1
TOP_predict = 1



def viterbi(k, transition, emission, words):
    n = len(words)
    path_dict = {}

    for i in range(n):
        path_dict[i] = {}
    for layer in path_dict:
        for state in states[1:]:
            path_dict[layer][state] = Buffer(k)

    path_dict[n] = {'stop': Buffer(k)}

    for layer in path_dict:
        # loop until the last one
        if layer == n:
            for previous_state in states[1:]:
                for k_th in range(k):
                    p = path_dict[layer - 1][previous_state].getProb(k_th) + \
                        transition[previous_state][current_state]['NULL']
                    path_dict[layer]['stop'].push(p, previous_state, k_th)
            continue
        #loop tgrough the state except the start and stop
        if layer == 1:
            for previous_state in states[1:]:
                for current_state in states[1:]:
                    for k_th in range(k):
                        p = path_dict[layer - 1][previous_state].getProb(k_th) + \
                            transition['NULL'][previous_state][current_state] + \
                            emission[words[layer]][current_state]
                        path_dict[layer][current_state].push(
                            p, previous_state, k_th)
            continue
        # at the begining
        if layer == 0:
            for current_state in states[1:]:
                for k_th in range(k):
                    p = transition['NULL']['NULL'][current_state] + \
                        emission[words[layer]][current_state]
                    path_dict[layer][current_state].push(p, "NA", k_th)
            continue

        for previous_state in states[1:]:
            for current_state in states[1:]:
                for k_th in range(k):
                    p = path_dict[layer - 1][previous_state].getProb(k_th) + \
                        transition[path_dict[layer - 1][previous_state].getPre_state(k_th)][previous_state][
                            current_state] + \
                        emission[words[layer]][current_state]
                    path_dict[layer][current_state].push(
                        p, previous_state, k_th)
    # since we only want the most suitiable sequence.
    top = 1
    current_layer = n - 1
    from_k_th = top - 1
    path_reverse = ["stop"]
    while current_layer >= 0:
        while path_dict[current_layer + 1][
                path_reverse[len(path_reverse) -
                             1]].getBuffer()[from_k_th]['pre_state'] == "NA":
            from_k_th -= 1
        path_reverse.append(
            path_dict[current_layer +
                      1][path_reverse[len(path_reverse) -
                                      1]].getBuffer()[from_k_th]['pre_state'])
        from_k_th = path_dict[current_layer + 1][path_reverse[
            len(path_reverse) - 2]].getBuffer()[from_k_th]['from_k_th']

        current_layer -= 1
    return path_reverse[::-1][:len(path_reverse) - 1]


def perceptron(tag_predictions, tags, words, transition, emission):
    n = len(tags)
    for i in range(n):
        emission[words[i]][tags[i]] += 1
        emission[words[i]][tag_predictions[i]] -= 1
        if i == 0:
            #0-start-next state
            transition["NULL"]["NULL"][tags[0]] += 1
            transition["NULL"]["NULL"][tag_predictions[0]] -= 1
        elif i == 1:
            #-start-next-state
            transition["NULL"][tags[0]][tags[1]] += 1
            transition["NULL"][tag_predictions[0]][tag_predictions[1]] -= 1
        else:
            transition[tags[i - 2]][tags[i - 1]][tags[i]] += 1
            transition[tag_predictions[i - 2]][tag_predictions[i - 1]][
                tag_predictions[i]] -= 1

    return transition, emission


# need to consider different marks
def remove_marks(word, clean=True):
    if clean:
        if word[:7] == 'http://':
            return "URL"
        return word
    else:
        return word

if __name__ == "__main__":
    # start the main process here
    mode_list = ['EN', 'test']
    if len(sys.argv) < 2:
        print ('Please make sure you have installed Python 3.4 or above!')
        print ("python part5.py <N>")
        print ("N=0: EN; N=1: test")
        sys.exit()
    mode = int(sys.argv[1])
    if mode < 0 or mode > 1:
        print ("Please input valid mode number!") 
        sys.exit()
    language = mode_list[mode]
    # can add different file folder to the list
    # need to change the language list
    print(language)
    train_file = open(language +
                    "/train",
                    encoding='utf8')
    # when the test file is coming will update here
    if mode==1:
        test_file = open(language + "/test.in",
                        encoding='utf8')
    else:
        test_file = open(language + "/dev.in",
                        encoding='utf8')

    transition = {}
    for state in states:
        transition[state] = {}
        for state_2 in states:
            transition[state][state_2] = {}
            for state_3 in states:
                transition[state][state_2][state_3] = 0

    lines = train_file.readlines()

    emission = {}
    for line in lines:
        if line != "\n":
            word = line.split(" ")[0]
            if word not in emission.keys():
                emission[remove_marks(word)] = {}
                for state in states[1:]:
                    emission[remove_marks(word)][state] = 0

    train_tag_data = [[]]
    train_word_data = [[]]
    cleaned_train_word_data = [[]]
    index = 0
    for line in lines:
        if line == '\n':
            index += 1
            train_tag_data.append([])
            train_word_data.append([])
            cleaned_train_word_data.append([])
            continue
        word = line.split(' ')[0]
        tag = line.split(' ')[1]
        train_tag_data[index].append(tag[:-1])
        train_word_data[index].append(word)
        cleaned_train_word_data[index].append(remove_marks(word))

    train_tag_data.pop()
    train_word_data.pop()
    cleaned_train_word_data.pop()

    # add new word
    lines = test_file.readlines()
    for line in lines:
        if line != "\n":
            word = line[:-1]
            if word not in emission.keys():
                emission[remove_marks(word)] = {}
                for state in states[1:]:
                    emission[remove_marks(word)][state] = 0

    test_word_data = [[]]
    cleaned_test_word_data = [[]]
    index = 0
    for line in lines:
        if line == '\n':
            index += 1
            test_word_data.append([])
            cleaned_test_word_data.append([])
            continue
        word = line[:-1]
        test_word_data[index].append(word)
        cleaned_test_word_data[index].append(remove_marks(word))

    test_word_data.pop()
    cleaned_test_word_data.pop()

    ##train
    for i in tqdm(range(iteration)):
        for j in range(len(train_tag_data)):
            prediction = viterbi(TOP_train, transition, emission,
                                cleaned_train_word_data[j])
            transition, emission = perceptron(prediction, train_tag_data[j],
                                            cleaned_train_word_data[j],
                                            transition, emission)
    #write each entry as a message
    message = ''
    for i in range(len(test_word_data)):
        prediction = viterbi(TOP_predict, transition, emission,
                            cleaned_test_word_data[i])
        for j in range(len(test_word_data[i])):
            message += test_word_data[i][j]
            message += ' '
            message += prediction[j]
            message += '\n'
        message += '\n'

    if mode==1:
        result = open(
            language +
            "/test.out", "wb")
    else:
        result = open(
            language +
            "/dev.p5.out", "wb")

    result.write(message.encode("utf-8"))
    result.close()

    print("Done")
