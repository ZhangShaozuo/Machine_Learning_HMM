# import part_4_buffer
from part_4_buffer import *
from HMM_part3 import *
top = 3

def top_k_sequence(k, test_sequence, tagOcc, tranTable, emiTable):
    state_without_start_stop = tagOcc.index.values
    length = len(test_sequence)
    tokens = emiTable.index.values

    # initializing the path_dic
    mybuffer = Buffer(k)
    mybuffer.push(1.0, 'NA', -1)
    path_dic = {0: {'START': mybuffer}}
    for i in range(length):
        # since we already pushed a dic to the path_dic already
        path_dic[i + 1] = {}
    for idx, element in enumerate(path_dic):
        if element == 0:
            continue
        for state in state_without_start_stop:
            path_dic[element][state] = Buffer(k)
    path_dic[length + 1] = {'STOP': Buffer(k)}
    for idx, element in enumerate(path_dic):
        # first step
        if element == 0:
            continue
        # last step
        if element == length + 1:
            for pre_state in path_dic[element - 1]:
                for k_th in range(k):
                    prob = path_dic[element - 1][pre_state].getProb(k_th) * tranTable['STOP'][pre_state]
                    path_dic[element]['STOP'].push(prob, pre_state, k_th)
            continue
        # loop the remaining state except the start and stop
        for state in path_dic[element]:
            for pre_state in path_dic[element - 1]:
                if test_sequence[element - 1] not in tokens:
                    for k_th in range(k):
                        prob = path_dic[element - 1][pre_state].getProb(k_th) *\
                            tranTable[state][pre_state] * emiTable[state]['#UNK#']
                        path_dic[element][state].push(prob, pre_state, k_th)
                else:
                    for k_th in range(k):
                        if emiTable[state][test_sequence[element - 1]]>0:
                            prob = path_dic[element - 1][pre_state].getProb(k_th) *\
                                tranTable[state][pre_state] * emiTable[state][test_sequence[element - 1]]
                        else:
                            # prob is almost 0
                            prob = path_dic[element - 1][pre_state].getProb(k_th) * tranTable[state][pre_state] * 0.00000000000001
                        path_dic[element][state].push(prob, pre_state, k_th)
    # backtracking all the state
    current_layer = length
    from_k_th = top - 1
    path_reverse = ['STOP']
    while current_layer >= 0:
        while path_dic[current_layer + 1][path_reverse[len(path_reverse) -1]].getBuffer()[from_k_th]['pre_state'] == 'NA':
            from_k_th -= 1
        path_reverse.append(path_dic[current_layer +1][path_reverse[len(path_reverse) -1]].getBuffer()[from_k_th]['pre_state'])
        from_k_th = path_dic[current_layer + 1][path_reverse[len(path_reverse) - 2]].getBuffer()[from_k_th]['from_k_th']
        current_layer -= 1
        # need to return the reverse of the path_reverse
    return path_reverse[::-1][1:len(path_reverse) - 1]

def evaluate_doc_part4(test, null_list, tranTable, emiTable, tagOcc, k):
    # test: test document without tag
    # null_list contains the index of tokens that are '#null#'
    tag = []
    print("Start to generate tag list:\n")
    # null_list=null_list[:2]
    for i in tqdm(range(len(null_list))):
        sequence = None
        # print(null_list[i])
        if (i == 0):
            sequence = test[:null_list[i]]
        else:
            sequence = test[null_list[i-1]+1:null_list[i]]
        tag.extend(top_k_sequence(k, sequence, tagOcc, tranTable, emiTable))
    return tag

def main_process(mode):
    trainpath = mode+'\\train'
    testpath = mode+'\\dev.in'
    predpath = mode+'\\dev.p4.predict'
    tranTable_savepath = mode+'\\devTrain\\tranTable.csv'
    emiTable_savepath = mode+'\\devTrain\\emiTable.csv'
    tag_savepath = mode+'\\tagp4.txt'
    data, tokenOcc, tagOcc = load_data(trainpath)
    tranTable = calCountTran(data, tagOcc)
    tranTable.to_csv(tranTable_savepath)
    emiTable = calCountEmi(data, tokenOcc, tagOcc)
    emiTable.to_csv(emiTable_savepath)

    test = load_data(testpath,False)
    null_list = fetch_space(test)
    tag_list = evaluate_doc_part4(test, null_list, tranTable, emiTable, tagOcc, top)
    # # save the tag in a file
    with open(tag_savepath, 'wb') as fp:
        pickle.dump(tag_list, fp)
    # with open(tag_savepath,'rb') as fp:
    #     tag_list=pickle.load(fp)
    write_pred(tag_list, testpath, predpath)


if __name__ == "__main__":
    print("Hello gemao")
    mode_list = ['EN', 'CN', 'SG']
    # # for i in range(len(mode_list)):
    # #     main_process(mode_list[i])
    main_process(mode_list[0])