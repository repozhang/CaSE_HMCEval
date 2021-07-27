
import pickle

class Csv_utils():
    def __int__(self):
        pass

    @staticmethod
    def get_number_list_dict(data,key,value):
        datadict={}
        for line in data:
            datadict.setdefault((line[key]), []).append(line[value])
        return datadict

    @staticmethod
    def save_pickle(data,filename):
        with open(filename, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_pickle(filename):
        with open(filename, 'rb') as f:
            out = pickle.load(f)
            return out



class Dict_utils():
    def __int__(self):
        pass

    @staticmethod
    def get_key_sum(mydict, value):
        return [k for k, v in mydict.items() if sum([int(i) for i in v]) == value]

    @staticmethod
    def get_key(mydict, value):
        return [k for k, v in mydict.items() if v == value]


if __name__=="__main__":
    testdict1 = {'hi': ['1', '2']}
    out=Dict_utils.get_key_sum(testdict1,3)
    print(out)