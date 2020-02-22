import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

class SVM:
    def __init__(self):
        self.colors = {1:'r',-1:'b',0:'g'}
        self.labels = {'train1':'train data(class 1)','train2':'train data(class 2)',
                        'test1':'test data(class 1)','test2':'test data(class 2)',
                        'decision_line':'decision boundry'}
        self.sizes = {'train':20,'test':20,'line':1}

            
    def fit(self,data):
        self.data = data
        norms_measure = {}
        transforms = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)   
        self.max_feature_value , self.min_feature_value= max(all_data) , min(all_data)
        all_data = None
        step_sizes = [self.max_feature_value * 0.1,self.max_feature_value * 0.01,self.max_feature_value * 0.001,]
        b_range_multiple , b_multiple= 1 , 5
        curr_min = self.max_feature_value
        for step in step_sizes:
            w = np.array([curr_min,curr_min])
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),self.max_feature_value*b_range_multiple,step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found = True
                        for yi in self.data:
                            for xi in self.data[yi]:
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    found = False        
                        if found:
                            norms_measure[np.linalg.norm(w_t)] = [w_t,b]
                if w[0] < 0:
                    optimized = True
                else:
                    w = w - step
            norms = sorted([n for n in norms_measure])
            min_norm = norms_measure[norms[0]]
            self.w,self.b= min_norm[0],min_norm[1]
            curr_min = min_norm[0][0]
        print(self.w,self.b)

    def predict(self,X):

        y_pred = np.sign(np.dot(np.array(X),self.w)+self.b)
        test_dict = {-1:[],1:[]}
        for group,test_data in zip(y_pred , X ):
            test_dict[group].append(test_data)
        test_dict[-1] , test_dict[1] = np.array(test_dict[-1]) , np.array(test_dict[1])
        plt.scatter(test_dict[-1][:,0],test_dict[-1][:,1],s=self.sizes['test'],color=self.colors[-1],marker='+',label=self.labels['test1'])
        plt.scatter(test_dict[1][:,0],test_dict[1][:,1],s=self.sizes['test'],color=self.colors[1],marker='+',label=self.labels['test2'])
        return y_pred

    def visualize(self):

        plt.scatter(data_dict[-1][:,0],data_dict[-1][:,1],s=self.sizes['train'],color=self.colors[-1],label=self.labels['train1'])
        plt.scatter(data_dict[1][:,0],data_dict[1][:,1],s=self.sizes['train'],color=self.colors[1],label=self.labels['train1'])
        plt.xlabel('x - axis') 
        plt.ylabel('y - axis') 
        plt.title('support vector machine')
        x_min , x_max = self.min_feature_value*0.9 , self.max_feature_value*1.1
        plt.plot([x_min,x_max],[(-self.w[0]*x_min-self.b) / self.w[1],(-self.w[0]*x_max-self.b) / self.w[1]],color=self.colors[0],label =self.labels['decision_line'])
        plt.legend()
        plt.show()


def get_data_from_file(filename):
    datafile = open(filename)
    data = []
    for row in datafile:
        tup = []
        for ele in row.split(','):
            tup.append(float(ele))
        data.append(np.array(tup))
    return np.array(data)


#feathing data from the file
train_data = get_data_from_file('./svm_train_data.txt')
test_data = get_data_from_file('./svm_test_data.txt')

#data preprocessing
data_dict = {-1:[],1:[]}
for tup in train_data:
    data_dict[tup[2]].append([tup[0],tup[1]])
data_dict[-1] , data_dict[1] = np.array(data_dict[-1]) , np.array(data_dict[1])

#training the model
clf = SVM()
clf.fit(data=data_dict)

#predicton
clf.predict(test_data)

#data visualization
clf.visualize()
