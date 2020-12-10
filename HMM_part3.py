from os.path import dirname
import numpy as np
from numpy.core.defchararray import count, index
from numpy.core.fromnumeric import argmax
from numpy.core.numeric import load
import pandas as pd
import os
from tqdm import tqdm
import pickle


def load_data(filepath, tag=True):
    dirpath = os.getcwd()  # Assume we are under the 'Project' directory
    filepath = os.path.join(dirpath, filepath)
    # load the data in the format of pandas datafrmae, skip blank lines are important
    # because the sequence is split by '\n'
    if (tag):
        print("Loading training dataset: ")
        data = pd.read_csv(filepath, header=None, delimiter=" ", names=[
            'token', 'tag'], skip_blank_lines=False)
        # Fill the empty token with '#null#' so that we can access it by data['token'].index
        data['token'] = data['token'].fillna('#null#')

        # tag Occurence has two columns:
        #   1st column is the unique tag name,
        #   2nd column is # of times this tag occurs in the document
        tagOcc = data['tag'].value_counts()

        # Same for token Occurence, take note this dataframe has '#null#'
        tokenOcc = data['token'].value_counts()
        return data, tokenOcc, tagOcc
    else:
        print("Loading test dataset")
        lines = None
        with open(filepath,'r',encoding='utf8') as fp:
            lines = fp.readlines()
        # print(len(lines))
        for i in range(len(lines)):
            lines[i] = lines[i][:-1]
        return lines


def calCountTran(data, tagOcc):
    numOfUniTag = tagOcc.index.values.shape[0]
    # After being inserted, NaArr would have all the possible tag
    # including 'START' and 'STOP'
    NaArr1 = np.insert(tagOcc.index.values, 0, 'START')
    NaArr = np.insert(NaArr1, NaArr1.shape[0], 'STOP')

    # Initialize the transition array with the shape below
    countArr = np.zeros((numOfUniTag+2, numOfUniTag+2), dtype=int)
    # Initialize the transition table with index and columns
    countTable = pd.DataFrame(countArr, columns=NaArr, index=NaArr)
    # manaully add Count(firstTag, 'START')
    firstTag = data['tag'][0]
    countTable[firstTag]['START'] += 1
    # Take note the pandas format:
    # count(y_i-1,y_i) is equivalent to counTable[y_i(column)][y_i-1(row)]
    print("Start to generate transition table:\n")
    for i in tqdm(range(1, data.shape[0])):
        previousTag = data['tag'][i-1]
        currentTag = data['tag'][i]
        # Recall that we didn't do data['tag'].fillna(something)
        # so there is NaN in data['tag'], we cannot access them,
        # but their types are float(where others are str)
        ci = (type(currentTag) == float)
        pi = (type(previousTag) == float)
        # Here we make use of NaN
        if (ci == False and pi == False):
            countTable[currentTag][previousTag] += 1
        if (ci == True and pi == False):
            # if the current tag is NaN, it indicates the end of a sequence
            countTable['STOP'][previousTag] += 1
        if (ci == False and pi == True):
            # if the previous tag is NaN, it indicates the start of a sequence
            countTable[currentTag]['START'] += 1
        if (ci == True and pi == True):
            # if both conditions are true, the only possibility is that we have reached the end of the document
            continue
    # calculate count(y_i)
    nOccTable = countTable.sum(axis=1)
    # transistion parameter = count(y_i-1,y_i)/count(y_i)
    tranTable = countTable.divide(nOccTable, axis=0).fillna(0)
    return tranTable


def calCountEmi(data, tokenOcc, tagOcc, k=0.5):
    # tokenOcc index values is a ndarray containing all the possible unique tokens
    # Viterbi algo will fail if the test set contains the token that doesn't exist in the train set
    # insert '#UNK' is to solve that problem
    unk = np.insert(tokenOcc.index.values, 0, '#UNK#')
    numOfUniToken = unk.shape[0]
    numOfUniTag = tagOcc.index.values.shape[0]
    # initialize the emission array
    EmiArr = np.zeros((numOfUniToken, numOfUniTag), dtype=int)
    # initialize the emission table
    EmiTable = pd.DataFrame(EmiArr, columns=tagOcc.index.values, index=unk)
    # iterate all the token in the training data(definitely there are lots of duplicates)
    print("Start to generate emission table:\n")
    for i in tqdm(range(data.values.shape[0])):
        curToken = data['token'][i]
        # if current token we are looking at is '#null', we do nothing
        if (curToken == '#null#'):
            continue
        else:
            curTag = data['tag'][i]
            EmiTable[curTag][curToken] += 1
    # emission parameter = count(y->x)/count(y)
    emiTable = EmiTable.divide(tagOcc, axis=1).fillna(0)
    # After doing division, we seperarely calculate e('#UNK#'|y), k=0.5
    for i in tagOcc.index.values:
        emiTable[i]['#UNK#'] = k/(tagOcc[i]+k)
    # 'START' and 'STOP' definitely emit nothing, inserting them is merely for matrix operation
    emiTable.insert(0, '#START#', 0)
    emiTable.insert(numOfUniTag+1, '#STOP#', 0)
    return emiTable


def HMM(sequence, tranTable, emiTable, tokenOcc):
    allState = tranTable.columns.values
    # sequence is ONE sequence. For test dataset we will process n times viterbi(n= # of sequences)
    sequence.insert(0,'#START#')
    sequence.append('#STOP#')
    allToken = np.array(sequence)
    # Here we are following the score table format taught in class
    # START--> Token 1--> Token2--> ... --> STOP
    # START, STOP act as fake tokens, they are merely for constructing scoretable purpose.
    rowLen = allState.shape[0]
    colLen = allToken.shape[0]
    # Initialize the score array
    scoreArr = np.zeros((rowLen, colLen))
    # Initialize the score table
    scoreTable = pd.DataFrame(scoreArr, index=allState, columns=allToken)
    # Iterate all the tokens in the test dataset
    for i in range(colLen):
        if i == 0:
            # i==0 means current token is START(fake token)
            # pi(0, START) = 1
            scoreTable.iloc[:, i] = scoreTable.iloc[:, i].add(1)
        elif i == colLen-1:
            # i==colLen-1 means current token is STOP(fake token)
            # pi(n+1,STOP) = max_v(pi(n,v)xcount(v,STOP))
            # For scoretable format, we doesn't have to take max
            scoreTable.iloc[:, i] = scoreTable.iloc[:,i-1].mul(tranTable['STOP'])
        else:
            curToken = allToken[i]
            # check whether curToken exists in training token set
            exists = curToken in tokenOcc.index.values
            # if doesn't exist, rename it to '#UNK' to resolve KeyError exception
            if (exists == False):
                curToken = '#UNK#'
            # Guarantine that curToken could be found in emission Table
            for uindex in range(rowLen):
                u = allState[uindex]
                # pi(i, u) = max_v(pi(i-1,v)xcount(v,u))xe(curToken|u)
                temp = scoreTable.iloc[:, i-1].mul(tranTable[u])
                # by utilizing matrix operation, we don't have to fill the score table cell by cell
                # but column by column
                if (u=='START'):
                    u='#START#'
                if (u=='STOP'):
                    u='#STOP#'
                scoreTable.iloc[uindex, i] = temp.max()*emiTable[u][curToken]
    return scoreTable


def evaluate_doc(test, null_list, tranTable, emiTable, tokenOcc):
    # test: test document without tag
    # null_list contains the index of tokens that are '#null#'
    tag = []
    print("Start to generate tag list:\n")
    # null_list=null_list[:2]
    for i in tqdm(range(len(null_list))): #len(null_list)
        sequence = None
        if (i == 0):
            sequence = test[:null_list[i]]
        else:
            sequence = test[null_list[i-1]+1:null_list[i]]
        scoreTable = HMM(sequence, tranTable, emiTable, tokenOcc)
        # generate the tag for this sequence
        tag_seq = evaluate_seq(sequence, scoreTable, tranTable)
        tag.extend(tag_seq)
    return tag


def evaluate_seq(sequence, scoreTable, tranTable):
    sequence.remove('#START#')
    sequence.remove('#STOP#')
    sequenceArr = np.array(sequence)
    # print("evaluate_seq:", sequenceArr.shape)
    tag = []
    # backtracking the scoretable to get the tag
    for i in range(sequenceArr.shape[0]-1, -1, -1):
        temp = None
        temp1 = None
        if i == sequenceArr.shape[0]-1:
            # last token: According to the design of scoretable,
            # take argmax of score['STOP'] would return us the tag with highest probability
            temp = scoreTable['#STOP#'].iloc[1:scoreTable['#STOP#'].shape[0]-1]
            tag.append(temp.idxmax())
        else:
            # Assume a seuqence:  #START# a b c #STOP#
            # Score table index:      0   1 2 3   4
            # Sequence index:        --   0 1 2   --
            # That's why for index i, we need to look for scoreTable.iloc[:,i+1]
            # In order to determine whether a tag u is the most probable tag
            # given the score from START to pi(i+1,u) and the next most probable tag is tag[-1]
            # we have to see whether pi(i+1,u)xcount(u,tag[-1]) give us the maximum value among all possible u
            # We don't have to multiple e(i+1|tag[-1]) because its value is consistent for all possible u
            temp = scoreTable.iloc[:, i+1].mul(tranTable[tag[-1]])
            temp1 = temp.iloc[1:temp.shape[0]-1]
            # temp.iloc[0]='START' and temp.iloc[-1]='STOP'
            # there is no way a token is tagged as 'START' and 'STOP'
            # for safety, we don't consider them when doing argmax.
            tag.append(temp1.idxmax())
    # reverse the tag since it's generated backwards
    tag.reverse()
    return tag


def fetch_space(data):
    # data is a pandas dataframe
    # the function is to identify the index of the tokens that are '#null#'
    # It helps us to split the test document to sequences
    null_list = []
    for i in tqdm(range(len(data))):
        if data[i] == '':
            null_list.append(i)
    # print(null_list)
    return null_list


def write_pred(tag_list, inpath='EN\\dev.in', predpath='EN\\dev.predict'):
    lines = []
    with open(inpath, 'r', encoding='utf8') as fp:
        lines = fp.readlines()
        # for line in lines:
        count = 0
        for i in range(len(lines)):
            if lines[i] == '\n':
                continue
            else:
                lines[i] = lines[i][:-1]+' '+tag_list[count]+'\n'
                count += 1
    with open(predpath, 'w', encoding='utf8') as fp:
        fp.writelines(lines)


def main_process(mode):
    trainpath = mode+'\\train'
    if mode == 'test':
        testpath = mode+ '\\test.in'
    else:
        testpath = mode+'\\dev.in'
    output_dir = mode+'\\part3'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    predpath = output_dir+'\\dev.p3.out'
    tranTable_savepath = output_dir+'\\tranTable.csv'
    emiTable_savepath = output_dir+'\\emiTable.csv'
    tag_savepath = output_dir+'\\tag.txt'
    data, tokenOcc, tagOcc = load_data(trainpath)
    tranTable = calCountTran(data, tagOcc)
    tranTable.to_csv(tranTable_savepath)
    emiTable = calCountEmi(data, tokenOcc, tagOcc)
    emiTable.to_csv(emiTable_savepath)

    test = load_data(testpath,False)
    null_list = fetch_space(test)
    tag_list = evaluate_doc(test, null_list, tranTable,
                            emiTable, tokenOcc)
    # # save the tag in a file
    with open(tag_savepath, 'wb') as fp:
        pickle.dump(tag_list, fp)
    ## If you don't want to train again, use these 2 lines to load the tag_list 
    # with open(tag_savepath,'rb') as fp:
    #     tag_list=pickle.load(fp)
    write_pred(tag_list, testpath, predpath)


if __name__ == "__main__":
    print("Hello World")
    mode_list = ['EN', 'CN', 'SG','test']
    main_process(mode_list[3])
    # for i in range(len(mode_list)):
    #     main_process(mode_list[i])


