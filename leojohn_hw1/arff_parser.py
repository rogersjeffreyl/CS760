from collections import defaultdict
from collections import OrderedDict

import re
class ARFF_Parser:
    def __init__(self):
        self.attribute_val_map = OrderedDict()
        self.attribute_type_map = defaultdict()
        self.type_attribute_map = defaultdict(list)
        self.attributes =[]
        self.class_attribute ="class"
        self.data=[]
    def  parse(self, file_name):
        self.data=[]
        start_of_data = False
        with open(file_name) as arff_file:
             for line in arff_file:
                line=line.strip()
                if start_of_data:
                    line_data = line.split(',')
                    attributes= self.attribute_val_map.keys()
                    curr_data =[]
                    for i in xrange(len(attributes)):
                        attr_name = attributes[i] 
                        if self.attribute_type_map[attr_name] == "numeric":
                            #if attr_name in  self.type_attribute_map["real"]:
                                curr_data.append((attributes[i],float(line_data[i])))
                            #elif attr_name in  self.type_attribute_map["numeric"]:
                                #curr_data.append((attributes[i], int(line_data[i])))
                        else: 
                            curr_data.append((attributes[i],line_data[i]))                
   
                    self.data.append(dict(curr_data) )   
    
                else:
                    match = re.match("@attribute\s+.*?((?:\w+\s*)+).*?\s+(.*)"\
                                     ,line)

                    if match !=None:
                        attr_name = match.group(1)
                        if attr_name != self.class_attribute:
                            self.attributes.append(attr_name)
                        type_value = match.group(2)
                        type_value =re.sub("(\w+)","'\\1'" \
                                           ,type_value)
                        type_value = eval(type_value.replace("{","[")\
                                      .replace("}","]") )

                        self.attribute_val_map[attr_name] = type_value
                        if type(type_value)== list:
                            self.type_attribute_map["nominal"].append(attr_name)
                            self.attribute_type_map[attr_name] = "nominal"
                        elif type_value =="real":
                            self.type_attribute_map["real"].append(attr_name)
                            self.attribute_type_map[attr_name] = "numeric"
                        elif type_value =="numeric":
                            self.type_attribute_map["numeric"].append(attr_name)
                            self.attribute_type_map[attr_name] = "numeric"
                    else:
                        if line =="@data":
                            start_of_data =  True
