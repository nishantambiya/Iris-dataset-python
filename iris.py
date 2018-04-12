#import iris dataset from sklearn.datasets
from sklearn.datasets import load_iris

#import decision tree classifier
from sklearn import tree

#Load iris datasets in iris
iris=load_iris()

#display features names 
print(iris.feature_names)

#display target classes
print(iris.target_names)

#display 1st dataset among 150 dataset
print(iris.data[0])

#display 1st dataset target which will be 0(setosa)
print(iris.target[0])

#define tree
clf=tree.DecisionTreeClassifier()

#train tree on dataset
clf=clf.fit(iris.data,iris.target)

#to visualize tree,it creates tree.dot file
#tree graph can be seen at url (http://webgraphviz.com/)
tree.export_graphviz(clf,out_file='tree.dot')    

#testing the tree with some data
test = [[5.1, 3.5, 1.4, 0.2]]
prediction=clf.predict(test)
print("Tested data is",prediction)
