import csv
from tqdm import tqdm


"""csv"""
class CsvProcess:
    def __init__(self,readfile,savefile):
        self.readfile=readfile
        self.savefile=savefile
        self.read_delimiter='\t'
        self.save_delimiter='\t'
        self.len=1000000

    def read_csv_origin(self):
        csv_file = open(self.readfile)
        csv_reader = csv.reader(csv_file, delimiter=self.read_delimiter,quotechar=None)
        return csv_reader #return list of data

    def read_csv(self):
        csv_file = open(self.readfile)
        csv_reader = csv.reader(csv_file, delimiter=self.read_delimiter,quotechar=None)
        linelist=[]
        k=0
        for line in tqdm(csv_reader):
            k+=1
            if k<=self.len:
                linelist.append(line)
        return (linelist) #return list of data

    @staticmethod
    def save_csv_origin(savefile,mydelimiter):
        file=open(savefile, 'a')
        writer = csv.writer(file, delimiter=mydelimiter, quoting=csv.QUOTE_MINIMAL)
        return writer


    def save_csv(self):
        savefile=open(self.savefile, 'a')
        writer = csv.writer(savefile, delimiter='\t', quoting=csv.QUOTE_NONE)
        for line in self.read_csv():
            # print(line)
            writer.writerow(line)


    @staticmethod
    def save_csv_static(list_to_save,save_file,sep_by,headername):
        savefile = open(save_file, 'a')
        writer = csv.writer(savefile, delimiter=sep_by, quoting=csv.QUOTE_NONE)
        i=0
        writer.writerow(headername)
        for index,line in enumerate(list_to_save):

            writer.writerow(line)
        print(i)

    @staticmethod
    def save_csv_static_line(line_to_save, save_file, sep_by):
        savefile = open(save_file, 'a')
        writer = csv.writer(savefile, delimiter=sep_by, quoting=csv.QUOTE_NONE)
        writer.writerow(line_to_save)





"""dictcsv"""
class CsvProcessDict:
    def __init__(self,readfile,savefile):
        self.readfile=readfile
        self.savefile=savefile
        self.delimiter='\t'
        self.fieldnames=None


    def read_csv_dict(self):
        csv_file=open(self.readfile)
        csv_reader=csv.DictReader(csv_file,delimiter=self.delimiter)
        return csv_reader


    def read_csv_dict_var(self,var_name):
        var_column_list=[]
        for line in self.read_csv_dict():
            var_column_list.append(line[var_name])
        return(var_column_list,len(var_column_list))

    @staticmethod
    def save_csv_dict_origin(savefile,fieldnames,delimiter):
        save_file=open(savefile,'a')
        csv_writer = csv.DictWriter(save_file, fieldnames=fieldnames, delimiter=delimiter)
        return(csv_writer)

    def save_csv_dict_line(self,line):
        save_file=open(self.savefile,'a')
        csv_writer = csv.DictWriter(save_file, fieldnames=self.fieldnames, delimiter=self.delimiter)
        csv_writer.writerow(line)