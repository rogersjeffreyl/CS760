from collections import OrderedDict
import itertools
from collections import defaultdict
import numpy as np
from collections import OrderedDict
import itertools
import pandas as pd
class NaiveBayes(object):
    def __init__(self,train_parser,classification_type="n"):
        self.laplace_class_counts = {}
        self.prior_probabilities={}
        self.conditional_probabilities = {}
        self.joint_probabilities = {}
        self.conditional_laplacian_counts = {}
        self.dependent_attribute =None
        self.dependent_attribute_values = None
        self.root_node = None
        self.feature_attributes =None
        self.conditional_mutual_info = defaultdict()
        self.graph=OrderedDict()
        self.tan_conditional_counts ={}
        self.tan_conditional_probabilities ={}
        self.classification_type =classification_type
        self.train_parser = train_parser    
        self.accuracy_score = 0

            
    def calculate_conditional_mutual_information(self):
        needed_columns =self.feature_attributes 
        for index_1 in xrange(0,len(needed_columns)) :
            for index_2 in xrange(0,len(needed_columns)) :

                    column_1= needed_columns[index_1]
                    column_2= needed_columns[index_2]

                    self.conditional_mutual_info[(column_1,column_2)]=0
                    for col_val_1 in self.train_parser.attribute_val_map[column_1]:
                        for col_val_2 in self.train_parser.attribute_val_map[column_2]:
                            for class_val in self.train_parser.attribute_val_map[self.dependent_attribute]:    
                                self.conditional_mutual_info[(column_1,column_2)] +=\
                                self.joint_probabilities[(column_1,column_2)][(col_val_1,col_val_2,class_val)]*\
                                np.log2(\
                                float(self.conditional_probabilities[(column_1,column_2)][(col_val_1,col_val_2,class_val)])/\
                                (self.conditional_probabilities[column_1][(col_val_1,class_val)]*self.conditional_probabilities[column_2][(col_val_2,class_val)])
                                )

                
    def prims(self):
        costs = [0*len(self.feature_attributes)]
        nodes = self.feature_attributes
        self.root_node = nodes[0]

        queue=nodes[1 :]
        #print "***"
        #print queue
        #print self.root_node
        #print "***"
        mst_edges =[]
        mst_edge_indices =[]
        nodes_in_mst=[self.root_node]
        while (len(queue)>0):
            max_wt=-1000
            max_to_node =None
            max_from_node = None
            max_index =0
            for node_1 in nodes_in_mst:
                for index,node_2 in enumerate(queue):

                    if self.conditional_mutual_info[(node_1,node_2)]>max_wt:
                        max_wt = self.conditional_mutual_info[(node_1,node_2)]
                        max_to_node = node_2
                        max_from_node = node_1
                        max_index = index
            nodes_in_mst.append(max_to_node)
            mst_edges.append((max_from_node,max_to_node))
            mst_edge_indices.append((nodes.index(max_from_node),\
                                   nodes.index(max_to_node)))
            
            queue.remove(max_to_node)
        return {"vertices":nodes_in_mst,"edges":mst_edges}
        
        

    def calculate_tan_conditional_probabilities(self,training_data):
        train_data = pd.DataFrame(training_data)
        
        for node in self.graph:
            column_class_count_dict = \
            train_data.groupby([node]+self.graph[node]).size().to_dict()
            column_values =[]
            for column in [node]+self.graph[node]:
                column_values.append(self.train_parser.attribute_val_map[column])
            keys = list(itertools.product(*column_values))
            for key in keys:
                if key in column_class_count_dict:
                    column_class_count_dict[key]+=1
                else:
                    column_class_count_dict[key]=1
            final_key = tuple([node]+self.graph[node])
            
            self.tan_conditional_counts[final_key] = column_class_count_dict
  
        for node in self.graph: 
            final_key = tuple([node]+self.graph[node])
            column_values=[]

            for column in self.graph[node]:
                column_values.append(self.train_parser.attribute_val_map[column])
            keys = list(itertools.product(*column_values))

            conditional_probability_table=defaultdict()
            for key in keys:
                total=0
                
                for val in self.train_parser.attribute_val_map[node]:
                    new_key =(val,)+key
                    total = total+\
                             self.tan_conditional_counts[final_key][new_key]
                        
                for val in self.train_parser.attribute_val_map[node]: 
                    new_key =(val,)+key
                    conditional_probability_table[new_key]=\
                    self.tan_conditional_counts[final_key][new_key]/float(total)
                    self.tan_conditional_probabilities[final_key]=conditional_probability_table
            
            
    def fit(self,train_data):

        self.dependent_attribute =self.train_parser.dependent_attribute
        self.feature_attributes = [attr \
                                   for attr in self.train_parser.attributes if attr != self.dependent_attribute]

        #Normal Conditional probabilities
        
        self.calculate_probabilities(train_data)    
        
        if self.classification_type == "t":
            self.calculate_conditional_mutual_information()
            result = self.prims()
            edges = result["edges"]
            vertices = result["vertices"]
            adjacency_matrix =defaultdict(list)
            for edge in edges:
                adjacency_matrix[edge[0]].append(edge[1])
            for vertex in vertices: 
                if vertex not in adjacency_matrix:
                    adjacency_matrix[vertex]=[]
   
            
            #Creating the graph of parent relationships
            parental_graph = defaultdict(list)
            for edge in edges:
                parental_graph[edge[1]].append(edge[0])
            for vertex in vertices: 
                if vertex not in parental_graph:
                    parental_graph[vertex]=[]  
            for node in self.feature_attributes:
                self.graph[node] =parental_graph[node]
                self.graph[node].append(self.dependent_attribute)
            for node in self.graph:    
                print " ".join([node]+self.graph[node]) 
            self.calculate_tan_conditional_probabilities(train_data)    
            
        else:
            for attr in self.feature_attributes:
                print " ".join([attr, self.dependent_attribute])
            
                
                   
    def calculate_probabilities(self,training_data):
        
        
        train_data =pd.DataFrame(training_data)
        self.dependent_attribute = self.train_parser.dependent_attribute
        train_sample_size = train_data.shape[0]
        self.dependent_attribute_values = self.train_parser.attribute_val_map[self.dependent_attribute]
        for values in train_data[self.dependent_attribute].unique():
            self.laplace_class_counts[values] =\
                train_data.groupby(self.dependent_attribute).size()[values]+1
        for values in train_data[self.dependent_attribute].unique():    
            self.prior_probabilities[values] =\
                float(self.laplace_class_counts[values])/\
                sum(self.laplace_class_counts.values())

                
        #Conditional laplacian counts
        for column in self.train_parser.attributes :
            #if column !=dependent_attribute:
                column_class_count_dict = train_data.groupby([column,self.dependent_attribute]).size().to_dict()
                for col_val in self.train_parser.attribute_val_map[column]:
                    for class_val in self.train_parser.attribute_val_map[self.dependent_attribute]:
                        if (col_val,class_val) not in column_class_count_dict:
                            column_class_count_dict [(col_val,class_val)]=1
                        else:
                            column_class_count_dict [(col_val,class_val)]+=1         

                self.conditional_laplacian_counts[column]=column_class_count_dict
        
        #Conditional laplacian counts for pairs of  attributes
        for column_1 in self.train_parser.attributes[:-1] :
            for column_2 in self.train_parser.attributes[:-1] :
                column_class_count_dict = train_data.groupby([column_1,column_2,self.dependent_attribute]).size().to_dict()
                for col_val_1 in self.train_parser.attribute_val_map[column_1]:
                    for col_val_2 in self.train_parser.attribute_val_map[column_2]:
                        for class_val in self.train_parser.attribute_val_map[self.dependent_attribute]:
                                if (col_val_1,col_val_2,class_val) not in column_class_count_dict:
                                    column_class_count_dict [(col_val_1,col_val_2,class_val)]=1
                                else:
                                    column_class_count_dict [(col_val_1,col_val_2,class_val)]+=1         

                self.conditional_laplacian_counts[(column_1,column_2)]=column_class_count_dict        
        #Conditional probabilities for  one attribute
        for column in self.train_parser.attributes :
            
                total = defaultdict()
                for class_val in self.train_parser.attribute_val_map[self.dependent_attribute]:
                    for col_val in self.train_parser.attribute_val_map[column]:
                        total[class_val] = total.get(class_val,0)+\
                                            self.conditional_laplacian_counts[column][(col_val,class_val)]
                conditional_probability_dict=defaultdict()
                for class_val in self.train_parser.attribute_val_map[self.dependent_attribute]:
                    for col_val in self.train_parser.attribute_val_map[column]:
                        conditional_probability_dict[(col_val,class_val)] = \
                            float(self.conditional_laplacian_counts[column][(col_val,class_val)])/total[class_val]
                self.conditional_probabilities[column]=conditional_probability_dict            
                
        #conditional probability distribution for pairs of attrinbutes
        needed_columns = self.feature_attributes
        for index_1 in xrange(0,len(needed_columns)) :
            for index_2 in xrange(0,len(needed_columns)) :

                    column_1= needed_columns[index_1]
                    column_2= needed_columns[index_2]
                    total= defaultdict()
                    for class_val in self.train_parser.attribute_val_map[self.dependent_attribute]:
                        total[class_val]=0
                        for col_val_1 in self.train_parser.attribute_val_map[column_1]:
                            for col_val_2 in self.train_parser.attribute_val_map[column_2]:
                                
                                total[class_val] = total.get(class_val,0)+\
                                                self.conditional_laplacian_counts[(column_1,column_2)][(col_val_1,col_val_2,class_val)] 
                    
                    conditional_probability_dict=defaultdict()
                    for class_val in self.train_parser.attribute_val_map[self.dependent_attribute]:
                        for col_val_1 in self.train_parser.attribute_val_map[column_1]:
                            for col_val_2 in self.train_parser.attribute_val_map[column_2]:
                                conditional_probability_dict[(col_val_1,col_val_2,class_val)] = \
                                    float(self.conditional_laplacian_counts[(column_1,column_2)][(col_val_1,col_val_2,class_val)])/total[class_val]
                    self.conditional_probabilities[(column_1,column_2)]= conditional_probability_dict  

                
        #joint probability distribution for pairs of attrinbutes

        needed_columns = self.feature_attributes
        for index_1 in xrange(0,len(needed_columns)) :
            for index_2 in xrange(0,len(needed_columns)) :

                    column_1= needed_columns[index_1]
                    column_2= needed_columns[index_2]
                    total = train_data.shape[0]+\
                    (len(self.train_parser.attribute_val_map[column_1])*\
                     len(self.train_parser.attribute_val_map[column_2])*\
                     len(self.train_parser.attribute_val_map[self.dependent_attribute])
                    )

                    joint_probability_dict=defaultdict()
                    for class_val in self.train_parser.attribute_val_map[self.dependent_attribute]:
                        for col_val_1 in self.train_parser.attribute_val_map[column_1]:
                            for col_val_2 in self.train_parser.attribute_val_map[column_2]:
                                joint_probability_dict[(col_val_1,col_val_2,class_val)] = \
                                    float(self.conditional_laplacian_counts[(column_1,column_2)][(col_val_1,col_val_2,class_val)])/total
                    self.joint_probabilities[(column_1,column_2)]= joint_probability_dict  
        
    def predict(self,test_data):
        print ""
        correctly_classified_instances = 0        
        if self.classification_type =='n':
            for data in test_data:
                final_score=defaultdict()

                for class_val in self.dependent_attribute_values:
                    final_score[class_val] = self.prior_probabilities[class_val]
                    for index in xrange(0,len(self.feature_attributes)):
                        attr_name = self.feature_attributes[index]
                        attr_value = data[index]
                        final_score[class_val] = final_score[class_val]*\
                                                 self.conditional_probabilities[attr_name][(attr_value,class_val)]
                max_score = None             
                final_class = None
                actual_class =None
                classes = final_score.keys()
                scores = final_score.values()
                max_score = max(scores)
                #print scores
                final_class = classes[scores.index(max_score)]
                if "'" in final_class:
                    final_class=eval(final_class)
                    actual_class=eval(data[-1])    
                else:
                    actual_class=data[-1]    
                
                if final_class == actual_class:
                    correctly_classified_instances +=1
                print "{0} {1} {2:.12f}".format(final_class,actual_class,float(max_score)/sum(final_score.values()))
            print "\n{0}".format(correctly_classified_instances)
            self.accuracy_score =  float(correctly_classified_instances)/test_data.shape[0]   
        else:            
            for data in test_data:
                final_score=defaultdict()
                for class_val in self.dependent_attribute_values:
                    final_score[class_val] = self.prior_probabilities[class_val]
                for node,parents in self.graph.iteritems():
                    final_key =  tuple([node]+parents)
                    cpt = self.tan_conditional_probabilities[final_key]
                    key=[]
                    #exclude the class_atrtribute
                    for column in [node]+parents[:-1]:
                        attr_index = self.feature_attributes.index(column)
                        attr_value = data[attr_index]
                        key.append(attr_value)

                    for class_val in self.dependent_attribute_values:
                        #Modify the key
                        new_key =tuple(key)+(class_val,)
                        final_score[class_val]=final_score[class_val]*cpt[new_key]
                max_score = None             
                final_class = None
                actual_class =None
                classes = final_score.keys()
                scores = final_score.values()
                max_score = max(scores)
                #print scores
                final_class = classes[scores.index(max_score)]
                if "'" in final_class:
                    final_class=eval(final_class)
                    actual_class=eval(data[-1])    
                else:
                    actual_class=data[-1]    
                if final_class == actual_class:
                    correctly_classified_instances +=1
                print "{0} {1} {2:.12f}".format(final_class,actual_class,float(max_score)/sum(final_score.values()))
            print "\n{0}".format(correctly_classified_instances)
            self.accuracy_score =  float(correctly_classified_instances)/test_data.shape[0]     