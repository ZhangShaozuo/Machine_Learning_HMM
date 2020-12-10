from HMM_part3 import load_data

if __name__=="__main__":
    trainpath = 'EN\\train'
    data, tokenOcc, tagOcc = load_data(trainpath)
    print(data.head())
    print(tagOcc.head())
    print(tagOcc+0.5)
