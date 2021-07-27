import pandas as pd
# import torch

def get_data(feature):
    input_file = '../data/data_per_dialogue.tsv'

    df = pd.read_csv(input_file, delimiter = "\t")

    original_headers = list(df.columns.values)

    numpy_array_train=df[df.apply(lambda x: x['my_split'] == 'train', axis=1)]
    numpy_array_test=df[df.apply(lambda x: x['my_split'] == 'test', axis=1)]

    ycol = ['WorkTimeInSeconds']

    Y_train=numpy_array_train[ycol]

    if feature=='f1':
        xcol=['turns', 'RewriteCount','FirstSubmit', 'DialogueLen', 'dia_dc_score', 'dia_fk_score', 'Mal_number', 'Nonmal_number', 'dialogue_bert_pp','Mal_label'] # f1
    else:
        xcol = ['turns', 'RewriteCount', 'FirstSubmit', 'DialogueLen', 'dia_dc_score', 'dia_fk_score', 'Mal_number',
                'Nonmal_number', 'dialogue_bert_pp', 'Mal_label','lifetime_ranking','worker_score']  # f2


    X_train=numpy_array_train[xcol]

    X_test = numpy_array_test[xcol]
    Y_test = numpy_array_test[ycol]
    # print(X_train,X_test)
    Y = df[ycol]
    X = df[xcol]

    return(X_train,Y_train,X_test,Y_test,X,Y)

if __name__=='__main__':
    feature='f2'
    X_train,Y_train,X_test,Y_test,X,Y=get_data(feature)
    print(X_train,Y_train,X_test,Y_test,X,Y)





