from collections import Counter
import math
import pandas as pd
import random
from collections import defaultdict
import numpy as np
class KNN:
    def __init__(self,train_data, test_data, parser):

        self.confusion_matrix =defaultdict(lambda: {})
        self.train_data = pd.DataFrame(train_data)
        self.test_data = pd.DataFrame(test_data)
        self.parser = parser
        self.accuracy = 0.0
        self.num_mis_classified = 0
        self.distance_matrix= None
        
    #Train and test data are pandas data frames   
    def __euclidean_distance__(self, point1,point2):
        
        np.sum(np.sqrt(np.square(np.subtract(point1,point2))))
    
    def fit(self):
        pass    
        #self.parser.parse(self.train_file)

        #rows = random.sample(range(0,len(self.parser.data)),int(float(self.train_percent)/100 * len(self.parser.data) ) )
        #needed_data = [self.parser.data[i] for i in rows]
        #self.train_data =pd.DataFrame(self.parser.data)

    def predict(self,k,print_output = True):
            correct_instances =0
            output_file = open("knn_output_{1}_{0}.txt".format(k,self.parser.dependent_attribute),'wb')
            output_file.write("k value : {0}\n".format(k))
            predicted = None
            actual = None
            error_calc = []
            #self.parser.parse(self.test_file)
            #self.test_data = pd.DataFrame(self.parser.data)
            self.train_data['dummy_join_key']=1
            self.test_data ['dummy_join_key']=1
            response_variable =self.parser.dependent_attribute
            feature_columns = set(self.train_data.columns).difference(set(['dummy_join_key',response_variable]))
            self.train_data.reset_index(inplace=True)

            self.test_data.reset_index(inplace=True)
            join=pd.merge(self.train_data,self.test_data,on="dummy_join_key")
            left_columns = [col+"_x" for col in feature_columns]
            right_columns = [col+"_y" for col in feature_columns]
            left_points = join[left_columns]
            right_points = join[right_columns]
            join['distance']= np.sqrt(np.sum(np.square(np.subtract(left_points,right_points)),axis=1))
            k_nearest_neighbors_data =join.groupby('index_y').apply(
                lambda dfg: dfg.nsmallest(k, columns=['distance','index_x'])
            )
            knnd =k_nearest_neighbors_data[['index_x','{0}_x'.format(self.parser.dependent_attribute),'distance','{0}_y'.format(self.parser.dependent_attribute)]]
            #Original response variables
            original_response_variables = self.parser.attribute_val_map[self.parser.dependent_attribute]
            for cls in original_response_variables:
                for cls1 in original_response_variables:
                    self.confusion_matrix[cls][cls1]=0
            for group, new_df in knnd.groupby(level=0):
                
                dataframe_dict = new_df.reset_index().to_dict()
                classes = dataframe_dict["{0}_x".format(self.parser.dependent_attribute)].values()
                actual = dataframe_dict["{0}_y".format(self.parser.dependent_attribute)].values()[0]
                if self.parser.dependent_attribute =="class":
                    class_counts = Counter(classes)
                    inv_map = {}
                    max_classes = None
                    for k, v in class_counts.iteritems():
                        inv_map.setdefault(v, []).append(k)
                        #print inv_map  
                    max_count = max(inv_map.keys())
                    max_classes = inv_map[max_count]
                    test_group =11
                    #if group == test_group:
                        #print  classes

                    #TIE IN THE CLASS PREDICTED BY NEAREST NEIGHBORS
                    if len(max_classes) >1:

                            indices =[original_response_variables.index(clas) for clas in max_classes]
                            min_index = min(indices)
                            if group ==test_group:
                                pass
                                #print max_classes
                                #print original_response_variables
                                #print  indices
                            #min_index_value = min(dataframe_dict["index_x"].values())
                            #predicted = self.train_data[self.train_data["index"]==min_index_value]["class"].values[0]
                            predicted = original_response_variables[min_index]

                    else:
                            predicted = max_classes[0] 

                    if actual == predicted:
                            error_calc.append(1)
                            self.confusion_matrix[actual][predicted]+=1
                    else:
                        self.num_mis_classified +=1
                        self.confusion_matrix[predicted][actual]+=1
                    #if group ==test_group:
                       #print predicted 
                    output_file.write("Predicted class : {0}	Actual class : {1}\n".format(predicted,actual))
                    if print_output:
                        print "Predicted class : {0}	Actual class : {1}".format(predicted,actual)
                else:

                    predicted = float(sum(classes))/k
                    output_file.write("Predicted value : {0:.6f}	Actual value : {1:.6f}\n".format(predicted,actual))
                    if print_output:
                        print "Predicted value : {0:.6f}	Actual value : {1:.6f}".format(predicted,actual)
                    error_calc.append(abs(predicted-actual))
            self.accuracy = float(sum(error_calc))/self.test_data.shape[0]       
            if self.parser.dependent_attribute =="class":
                output_file.write("Number of correctly classified instances : {0}\n".format(sum(error_calc)))
                if print_output:
                    print "Number of correctly classified instances : {0}".format(sum(error_calc))
                output_file.write("Total number of instances : {0}".format(self.test_data.shape[0]))
                if print_output:
                    print "Total number of instances : {0}".format(self.test_data.shape[0])
                
                output_file.write("Accuracy : {0:.16f}\n".format(self.accuracy) )        
                if print_output:
                    print "Accuracy : {0:.16f}".format(self.accuracy) 
            else:
                output_file.write("Mean absolute error : {0:.16f}\n".format(self.accuracy))
                if print_output:
                    print "Mean absolute error : {0:.16f}".format(self.accuracy)
                output_file.write("Total number of instances : {0}\n".format(self.test_data.shape[0]))
                if print_output:
                    print "Total number of instances : {0}".format(self.test_data.shape[0])

            output_file.close()
