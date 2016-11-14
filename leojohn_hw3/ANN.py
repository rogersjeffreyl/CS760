import numpy as np
import random
class ANN(object):
    #assume that the data is preprocessed by adding a bias term
    def __init__(self,h,X,y):
        self.h=h
        self.total_layers =2
        self.node_in_each_level=[X.shape[1]]
        self.cross_entropy_error =0
        self.weights=[]
        self.X=X
        self.y=y
        self.predited_confidence = []
        self.predicted_labels = []
        self.error_rate=0
        self.error_rate =0
        if h>0:
            self.total_layers +=1
            self.node_in_each_level.append(h)
        self.node_in_each_level.append(1)            
        for level in xrange(0,self.total_layers-1):
            lvl_1 = self.node_in_each_level[level]
            if level+1 == len(self.node_in_each_level)-1  and level >0:
                lvl_1 +=1
            lvl_2 = self.node_in_each_level[level+1]
            #self.weights.append( np.random.uniform(-0.01, 0.01,lvl_1*lvl_2).\
            #reshape(lvl_1, lvl_2))
            self.weights.append(np.array([random.uniform(-0.01, 0.01) for i in xrange(0,lvl_1*lvl_2)]).reshape(lvl_1, lvl_2))
        #print self.weights     
    def fit(self,l,e):
        
        correct_classified =0
        mis_classified =0
        for epoch in xrange(0,e) :
            self.cross_entropy_error=0
            correct_classified =0
            mis_classified =0
            for array_index in xrange(self.X.shape[0]):
                x= self.X[array_index]
                y= self.y[array_index]
                level_1_output =None
                level_2_output =None
                final_output = None
                if self.h>0:
                    level_1_output = 1/(1+np.exp(-(np.dot(x,self.weights[0]))))
                    #adding bias term for next layer
                    level_1_output_with_bias = np.insert(level_1_output, 0, 1)
                    level_2_output = 1/(1+np.exp(-(np.dot(level_1_output_with_bias,self.weights[1]))))
                    final_output = level_2_output[0]
                else:
                    level_1_output = (1/(1+np.exp(-(np.dot(x,self.weights[0]))))) 
                    final_output = level_1_output[0]
                hidden_unit_delta =None   
                #print final_output
                
                #y_num = np.where(np.unique(train_data[train_parser.dependent_attribute].values) == y)[0][0]
                output_unit_delta = y - final_output 
                
                if self.h>0:
                    hidden_unit_delta = level_1_output *(1-level_1_output)*output_unit_delta*self.weights[1]  
                    hidden_unit_delta = output_unit_delta*np.multiply( level_1_output *(1-level_1_output),(self.weights[1][1 :]).reshape(level_1_output.shape) )
                    #print hidden_unit_delta
                    self.weights[1] += l*output_unit_delta*level_1_output_with_bias.reshape(self.weights[1].shape)
                    self.weights[0]+=l*np.mat(x).T*np.mat(hidden_unit_delta)
                else:
                    #print output_unit_delta
                    #print x.shape
                    self.weights[0]+=(l*output_unit_delta*x).reshape(self.weights[0].shape)
                # classification step:
                outcome = 0
                if final_output >=0.5:
                    outcome= 1
                else:
                    outcome= 0
                if outcome == y:
                    correct_classified +=1  
                else:
                    mis_classified +=1
                self.cross_entropy_error=self.cross_entropy_error+ (-y*np.log(final_output)-((1-y)*np.log(1-final_output)))       
            print "Epoch:{0}\tCorrectly Classified:{1}\tMis-Classified:{2}\tCross-entropy-error:{3}".format(epoch+1,correct_classified,mis_classified,self.cross_entropy_error)
    def predict(self,test_X,test_y):
        self.predited_confidence = []
        self.predicted_labels= []
        self.error_rate=0
        level_1_output =None
        level_2_output =None
        final_output = None 
        correct_classified =0
        mis_classified =0
        for array_index in xrange(test_X.shape[0]):
            x= test_X[array_index]
            y= test_y[array_index]
            if self.h>0:
                level_1_output = 1/(1+np.exp(-(np.dot(x,self.weights[0]))))
                #adding bias term for next layer
                level_1_output_with_bias = np.insert(level_1_output, 0, 1)
                level_2_output = 1/(1+np.exp(-(np.dot(level_1_output_with_bias,self.weights[1]))))
                final_output = level_2_output[0]
            else:
                level_1_output = (1/(1+np.exp(-(np.dot(x,self.weights[0]))))) 
                final_output = level_1_output[0]
            outcome = 0
            if final_output >=0.5:
                    outcome= 1
            else:
                    outcome= 0
            self.predited_confidence.append(final_output)
            if outcome ==y:
                correct_classified +=1  
            else:
                mis_classified +=1
            self.predicted_labels.append(outcome)          
            print "Activation:{0}\tPredicted:{1}\tActual:{2}".format(final_output,outcome,y)        
        print "Correctly classified:{0}\tMis-classified:{1}".format(correct_classified,mis_classified)       
        self.error_rate=float(mis_classified)/(correct_classified+mis_classified)