from __future__ import print_function
from arff_parser import ARFF_Parser as parser 
import numpy as np
import scipy as sp
import pandas as pd
from collections import defaultdict
from collections import OrderedDict
import random
class TreeNode(object):
    def __init__(self, attribute, value,node_type,operator,parent,count_class1, count_class2  ):
        self.attribute = attribute
        self.value = value
        self.node_type = node_type
        self.operator = operator
        self.parent = parent
        self.count_class1 = int(count_class1)
        self.count_class2 = int(count_class2)
        self.children = []
        self.is_leaf = False
        self.leaf_class = None
    @staticmethod
    def display(node):
        if node == None:
            return
        TreeNode.display_rec(node, -1)
    
    @staticmethod    
    def display_rec( node,depth ):
        if node.attribute !="DUMMY" :
            TreeNode.display_curr_node(node, depth)
        for n in node.children:
                TreeNode.display_rec(n, depth+1)
    @staticmethod            
    def display_curr_node (node, depth):        
        print ("|\t"*depth, end='')
        if  node.node_type =="nominal":

            print (node.attribute.strip() + " = " + node.value + " [" + str(node.count_class2) \
                                 + " " + str(node.count_class1) + "]", end="")
        else:
            print( node.attribute +" "+ node.operator + " ", end="")
            print ('{:0.6f}'.format(node.value),end="")
            print (" [" + str(node.count_class2) + " " + str(node.count_class1) + "]",end="")
            
        if node.is_leaf:
            print (": "+ node.leaf_class, end="")
        print ("", end='\n')


class DecisionTree:
    def __init__(self,m,train_file, test_file, train_percent =100):
        self.m = m
        self.train_file =train_file
        self.test_file = test_file
        self.train_data = None
        self.test_data = None
        self.parser = parser()
        self.train_percent =train_percent
        self.accuracy = 0.0
        self.root = TreeNode("DUMMY", "DUMMY", "nominal","", None, 0 , 0)
    #Train and test data are pandas data frames   
    def __calculate_entropy__(self, pos, neg):
        if pos ==0 or neg==0:
            return 0.0
        p1 =float(pos)/(pos+neg)
        p2 =float(neg)/(pos+neg)
        return  -p1* np.log2(p1) -p2 * np.log2(p2)
    
    def fit(self):

        self.parser.parse(self.train_file)
        rows = random.sample(range(0,len(self.parser.data)),int(float(self.train_percent)/100 * len(self.parser.data) ) )
        needed_data = [self.parser.data[i] for i in rows]
        self.train_data =pd.DataFrame(needed_data)
        count_dict = self.train_data.groupby(["class"])["class"].count().to_dict()
        entropy = self.__calculate_entropy__ (count_dict.get(self.parser.attribute_val_map["class"][1],0.0), count_dict.get(self.parser.attribute_val_map["class"][0], 0.0))
        bitmap= [True]*self.train_data.shape[0]
        self.__candidate_split__(bitmap, entropy,  self.root)
        TreeNode.display(self.root)
        
    
    def predict(self):
        self.parser.parse(self.test_file)
        self.test_data = self.parser.data
        print ("<Predictions for the Test Set Instances>")
        num_correct =0
        for index, instance in enumerate(self.test_data):
            node = self.__find_first_node__(self.root,instance)
            if node == None:
                print ("Could not locate the node")
                return
            else:
                class_value = self.__check_instance__(node, instance)
                #TODO: Display data
                if instance["class"] == class_value:
                    num_correct = num_correct+1
                print("{0}: Actual: {1} Predicted: {2}".format(index+1,instance["class"],class_value))
        print ("Number of correctly classified: {0} Total number of test instances: {1}"\
               .format(num_correct,len(self.test_data)))
        self.accuracy =float( num_correct) /len(self.test_data)
    def __check_instance__(self,node,instance):
        if node==None:
            return None
        if node.is_leaf:
            return node.leaf_class
        for child in node.children:
            if child.node_type =="nominal":
                if instance [child.attribute] == child.value :
                    class_val = self.__check_instance__(child, instance)
                    if class_val !=None:
                        return class_val
            else:
                if child.operator =="<=":
                    if instance [child.attribute] <=child.value:
                        class_val = self.__check_instance__(child, instance)
                        if class_val !=None:
                            return class_val
                else:
                    if instance [child.attribute] >child.value:
                        class_val = self.__check_instance__(child, instance)
                        if class_val !=None:
                            return class_val
            
            
    def __find_first_node__(self,root,instance):
        children = root.children
        for child in children:
            if child.node_type =="nominal":
                if instance [child.attribute] == child.value :
                    return child
            else:
                if child.operator =="<=":
                    if instance [child.attribute] <=child.value:
                        return child
                else:
                    if instance [child.attribute] >child.value:
                        return child    
                
            
        return None    
        
    def __get_numeric_attributes_count_across_threshold__(self, attribute, threshold, bitmap):
        input_data = self.train_data[pd.Series(bitmap)]
        less_than_or_equals = input_data[input_data[attribute] <=threshold]
        greater_than = input_data[input_data[attribute] > threshold]
        lt_dict = less_than_or_equals.groupby('class')[attribute].count().to_dict()
        gt_dict =greater_than.groupby('class')[attribute].count().to_dict()
        for k in self.parser.attribute_val_map["class"]:

            if k not in lt_dict:
                lt_dict [k] =0.0
            if k not in gt_dict:
                gt_dict [k] =0.0
        return [lt_dict, gt_dict]        

    def __get_nominal_attributes_count__(self,attribute,bitmap):
        input_data = self.train_data[pd.Series(bitmap)]
        return input_data.groupby([attribute,'class'])[attribute].count().to_dict()
    
    def __calculate_info_gain_nominal__(self, attribute,entropy,bitmap):
        nominal_count = self.__get_nominal_attributes_count__(attribute,bitmap)


        nominal_count_dict = defaultdict()
        total = sum(nominal_count.values())

        attribute_entropy =[]
        conditional_entropy = 0.0
        #print nominal_count
        for k,v in nominal_count.iteritems():
            if k[0] not in nominal_count_dict:
                nominal_count_dict[k[0]]={k[1]:v}
            else:    
                nominal_count_dict[k[0]][k[1]]=v
        #print nominal_count_dict
        for k,v in nominal_count_dict.iteritems():
            #print k,v
            value_total = v.get(self.parser.attribute_val_map["class"][1],0.0) + v.get(self.parser.attribute_val_map["class"][0],0.0)
            #print "Value Total",value_total
            ratio = float(value_total)/ total
            #print "ratio",ratio
            attribute_entropy = ratio*self.__calculate_entropy__(v.get(self.parser.attribute_val_map["class"][1],0.0),v.get(self.parser.attribute_val_map["class"][0], 0.0))
            #print "entropy",self.__calculate_entropy__(v.get(self.parser.attribute_val_map["class"][1],0.0),v.get(self.parser.attribute_val_map["class"][0], 0.0))
            conditional_entropy += attribute_entropy
        #print "Attribute:{0},Total{4} Nominal Count:{1}, Entropy{2},Conditional:{3}".format(attribute,nominal_count,entropy,conditional_entropy,total)
        return entropy - conditional_entropy
    def __generate_thresholds__(self, attribute,bitmap):
        candidate_splits =[]
        input_data = self.train_data[pd.Series(bitmap)]
        train_data_groups = []
        sorted_attributes = np.sort(input_data[attribute].unique())
        for val in sorted_attributes:
            train_data_groups.append(self.train_data[self.train_data[attribute]== val])
        for i in xrange(0,len(train_data_groups)-1)  :
            if (self.parser.attribute_val_map["class"][1] in train_data_groups[i]['class'].unique() \
                and self.parser.attribute_val_map["class"][0] in train_data_groups[i+1]['class'].unique() )\
                or \
                (self.parser.attribute_val_map["class"][1] in train_data_groups[i+1]['class'].unique() \
                and self.parser.attribute_val_map["class"][0] in train_data_groups[i]['class'].unique() ):
                
                    candidate_splits.append(float(sorted_attributes[i]+sorted_attributes[i+1])/2) 
        return candidate_splits             
                    
    def __find_entropy_counts__(self,bitmap):

        input_data =self.train_data[pd.Series(bitmap)]
        return input_data.groupby(["class"])["class"].count().to_dict()
    def __calculate_info_gain_numeric__(self,attribute,entropy, bitmap):
        thresholds = self.__generate_thresholds__(attribute, bitmap)
        attr_entropy =[0.0]*len(self.parser.attribute_val_map["class"])
        max_threshold =0
        max_gain = 0
        #print thresholds
        for thresh in thresholds:

            count = self.__get_numeric_attributes_count_across_threshold__(attribute, thresh, bitmap)
    
            attr_entropy[0] = self.__calculate_entropy__ (count[0][self.parser.attribute_val_map["class"][1]], count[0][self.parser.attribute_val_map["class"][0]])       
            attr_entropy[1] = self.__calculate_entropy__ (count[1][self.parser.attribute_val_map["class"][1]], count[1][self.parser.attribute_val_map["class"][0]])       
            total = sum ([ count[i][class_val] for i in xrange(0,2) for class_val in [self.parser.attribute_val_map["class"][1], self.parser.attribute_val_map["class"][0]]])
            conditional_entropy = 0.0

            #conditional_entropy
            for i in xrange(0,2):
                local_total =count[i][self.parser.attribute_val_map["class"][1]]+ count[i][self.parser.attribute_val_map["class"][0]]

                ratio = float(local_total)/total
                conditional_entropy  = conditional_entropy+ ratio *attr_entropy[i]
            info_gain = entropy - conditional_entropy
            #print "Entropy is {0} conditional entropy is {1}".format(entropy, conditional_entropy)
            #print "INfo gain for {0} is {1}".format(thresh, info_gain)
            if info_gain >max_gain:
                max_gain = info_gain
                max_threshold = thresh 
        #print {"max_gain":max_gain, "max_threshold": max_threshold} 
        return {"max_gain":max_gain, "max_threshold": max_threshold}   
    
    
    
    def __find_parent_class__(self, node, bitmap):
        #counts =self.__find_entropy_counts__(bitmap)
        #if counts[self.parser.attribute_val_map["class"][0]] ==counts[self.parser.attribute_val_map["class"][1]]:
        #Note: IN the tree node class class1 is positive and class2 is negative 
        if node.count_class2==node.count_class1:
             return self.__find_parent_class__( node.parent, bitmap)
        #elif counts[self.parser.attribute_val_map["class"][1]] < counts[self.parser.attribute_val_map["class"][0]]:    
        elif node.count_class2 > node.count_class1: 
             return self.parser.attribute_val_map["class"][0]
        else:
            return self.parser.attribute_val_map["class"][1]
    def __set_leaf_node__(self,bitmap,parent):
        #print "set leafa node"
        counts =self.__find_entropy_counts__(bitmap)
        cls =None
        if counts[self.parser.attribute_val_map["class"][0]] ==counts[self.parser.attribute_val_map["class"][1]]:
            cls = self.__find_parent_class__(parent.parent,bitmap) 
        elif counts[self.parser.attribute_val_map["class"][1]] < counts[self.parser.attribute_val_map["class"][0]]:    
            cls=self.parser.attribute_val_map["class"][0]
        else:
            cls=self.parser.attribute_val_map["class"][1]
        parent.leaf_class = cls
        #print "settong true"
        parent.is_leaf = True
    def __update_bitmap__(self,bitmap,attribute,attr_type,nominal_value,
                         numeric_condition, threshold):

        input_data=self.train_data
        #print nominal_value
        if attr_type =="nominal":
            return np.logical_and((input_data[attribute] == nominal_value).as_matrix(),bitmap)
        elif attr_type =="numeric":
            #print threshold
            if(numeric_condition  =="<="):
                return np.logical_and((input_data[attribute] <= threshold).as_matrix(),bitmap) 
            else:
                return np.logical_and((input_data[attribute] > threshold).as_matrix(),bitmap)     
    def __determine_leaf_class__(self,entropy_counts):
        if entropy_counts.get(self.parser.attribute_val_map["class"][1],0.0) == entropy_counts.get(self.parser.attribute_val_map["class"][0],0.0):
            return self.parser.attribute_val_map["class"][0] 
        elif entropy_counts.get(self.parser.attribute_val_map["class"][1],0.0) == 0:
            return self.parser.attribute_val_map["class"][0]
        else:
            return self.parser.attribute_val_map["class"][1]
        
    def __candidate_split__(self,bitmap,entropy, parent):
        #print entropy
        #print bitmap
        
        found_pos= False
        max_gain=0.0
        current_threshold=0.0
        threshold = 0.0 
        candidate_attribute =""
        new_bitmap = [False]*len(bitmap)
        #print np.count_nonzero(bitmap)
        if np.count_nonzero(bitmap) < self.m:
                self.__set_leaf_node__(bitmap, parent);
                return

        for attribute in self.parser.attributes:
            if self.parser.attribute_type_map[attribute] \
               == "nominal":
                gain = self.__calculate_info_gain_nominal__( attribute, entropy, bitmap)
                
                if gain>0:
                    found_pos =True
            elif self.parser.attribute_type_map[attribute] \
                == "numeric":
                num_gain = self.__calculate_info_gain_numeric__( attribute, entropy, bitmap)
                gain = num_gain["max_gain"]
                threshold = num_gain["max_threshold"]
                if gain>0:
                    found_pos =True
            #print "Gain for attribute {0} is {1}".format(attribute,gain)
            if gain >max_gain:
                max_gain =gain
                current_threshold = threshold
                candidate_attribute = attribute
                
        #print "Max Gain:{0}".format(max_gain)
        #print "Attr:{0}".format(candidate_attribute)        
        if found_pos ==False:
        
            self.__set_leaf_node__(bitmap, parent)
            return
        
        if self.parser.attribute_type_map[candidate_attribute] =="nominal":
            for value in self.parser.attribute_val_map[candidate_attribute]:
                new_bitmap = self.__update_bitmap__(bitmap,candidate_attribute,"nominal",value,
                         "numeric_condition", 0.0)
                entropy_counts =self.__find_entropy_counts__(new_bitmap)  
                entropy = self.__calculate_entropy__(entropy_counts.get(self.parser.attribute_val_map["class"][1],0.0), entropy_counts.get(self.parser.attribute_val_map["class"][0],0.0))
                if entropy ==0.0:

                    class_val = self.__determine_leaf_class__(entropy_counts)
                    child = TreeNode(candidate_attribute, value,"nominal","",parent,entropy_counts.get(self.parser.attribute_val_map["class"][1],0.0), entropy_counts.get(self.parser.attribute_val_map["class"][0],0.0) )
                    parent.children.append(child)
                    child.leaf_class = class_val
                    child.is_leaf = True
                else:

                    child = TreeNode(candidate_attribute, value,"nominal","",parent,entropy_counts.get(self.parser.attribute_val_map["class"][1],0.0), entropy_counts.get(self.parser.attribute_val_map["class"][0],0.0) )
                    parent.children.append(child)
                    #print entropy,value
                    self.__candidate_split__(new_bitmap, entropy, child)
        else:
            #print "Numeric"

            new_bitmap = self.__update_bitmap__(bitmap,candidate_attribute,"numeric","",
                         "<=",current_threshold)

            entropy_counts =self.__find_entropy_counts__(new_bitmap)  
            entropy = self.__calculate_entropy__(entropy_counts.get(self.parser.attribute_val_map["class"][1],0.0), entropy_counts.get(self.parser.attribute_val_map["class"][0],0.0))
 
            if entropy ==0.0:
                    class_val = self.__determine_leaf_class__(entropy_counts)
                    child = TreeNode(candidate_attribute, current_threshold,"numeric","<=",parent,entropy_counts.get(self.parser.attribute_val_map["class"][1],0.0), entropy_counts.get(self.parser.attribute_val_map["class"][0],0.0) )
                    parent.children.append(child)
                    child.leaf_class = class_val
                    child.is_leaf = True

            else:
                    child = TreeNode(candidate_attribute, current_threshold,"numeric","<=",parent,entropy_counts.get(self.parser.attribute_val_map["class"][1],0.0), entropy_counts.get(self.parser.attribute_val_map["class"][0],0.0) )
                    parent.children.append(child)
                    self.__candidate_split__(new_bitmap, entropy, child)
            #print "______"
            #print np.where(new_bitmap)
            #print current_threshold
            new_bitmap = self.__update_bitmap__(bitmap,candidate_attribute,"numeric","",
                         ">",current_threshold)
            
            #print np.where(new_bitmap)
            entropy_counts =self.__find_entropy_counts__(new_bitmap)  
            entropy = self.__calculate_entropy__(entropy_counts.get(self.parser.attribute_val_map["class"][1],0.0), entropy_counts.get(self.parser.attribute_val_map["class"][0],0))
            #print "EEEE",entropy
            #print "______"
            if entropy ==0.0:
                    class_val = self.__determine_leaf_class__(entropy_counts)
                    child = TreeNode(candidate_attribute, current_threshold,"numeric",">",parent,entropy_counts.get(self.parser.attribute_val_map["class"][1],0), entropy_counts.get(self.parser.attribute_val_map["class"][0],0) )
                    parent.children.append(child)
                    child.leaf_class = class_val
                    child.is_leaf = True
            else:
                    child = TreeNode(candidate_attribute, current_threshold,"numeric",">",parent,entropy_counts.get(self.parser.attribute_val_map["class"][1],0), entropy_counts.get(self.parser.attribute_val_map["class"][0],0) )
                    parent.children.append(child)
                    self.__candidate_split__(new_bitmap, entropy, child)
