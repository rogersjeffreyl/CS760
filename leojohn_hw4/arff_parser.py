from collections import defaultdict
from collections import OrderedDict
import math
import re
import scipy
from scipy.io.arff import loadarff
class ARFF_Parser:
    def __init__(self):
        self.attribute_val_map = OrderedDict()
        self.attribute_type_map = defaultdict()
        self.type_attribute_map = defaultdict(list)
        self.attributes =[]
        self.dependent_attribute = None
        self.dependent_attribute_types =["class" , "response"]
        self.task_types ={"class": "classification" ,"response":"regression"}
        self.task_type = None
        self.data=[]
        
        
    def parse(self, file_name):
        loaded_data = loadarff(file_name)
        self.data = loaded_data[0]
        self.attributes =  loaded_data[1].names()
        self.attribute_type_map = dict(zip(loaded_data[1].names(),loaded_data[1].types()))
        for k,v in self.attribute_type_map.iteritems():
            self.type_attribute_map[v].append(k)
        for attr in loaded_data[1].names():
            if attr.lower() in self.dependent_attribute_types:
                self.dependent_attribute = attr.lower()
                self.task_type =self.task_types[attr.lower()]
                break
        #print loaded_data[1].__repr__()        
        nominal_attrs = re.findall(r"(.*)'s type is nominal,\s*range is (.*)",loaded_data[1].__repr__())     
        #print nominal_attrs
        for attrs,val in nominal_attrs:
            self.attribute_val_map[attrs.lstrip()] =   eval(val)
        