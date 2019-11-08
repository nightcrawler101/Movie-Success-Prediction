import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree,metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import graphviz
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import accuracy_score
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd



def plot_cluster(clusters,centroids):
	for key,value in clusters.items():
		if value == 0:
			c = 'red'
			plt.scatter(*list(key), s = 20, c = c)
		elif value == 1:
			c = 'blue'
			plt.scatter(*list(key), s = 20, c = c)
		elif value == 2:
			c = 'green'
			plt.scatter(*list(key), s = 20, c = c)
		elif value == 3:
			c = 'magenta'
			plt.scatter(*list(key), s = 20, c = c)	
		elif value == 4:
			c = 'yellow'
			plt.scatter(*list(key), s = 20, c = c)
	plt.scatter(*zip(*centroids), s = 50, c = 'black')
	plt.show()


def formClusters(Centroids,dataset):
	cluster = {}
	for point in dataset:
		cdist = []
		for c in Centroids:
			dist = (sum([(x-y)**2 for x,y in zip(point,c)]))**0.5
			cdist.append(dist)
		#print("Distance vector is: ",cdist)
		cluster[point] = cdist.index(min(cdist))
	return cluster

def compCentroid(points,n):
	if points != []:
		centroid = []
		for i in range(n):
			val = sum([axis[i] for axis in points]) / len(points)
			centroid.append(val)
		return(tuple(centroid))

filename = "movie.csv"
attribute = []
data = []

countEmptyGross=0
countEmptyIMDBScore=0
countEmptyBudget=0
total_count_gross=0
total_count_imdb=0
total_count_budget=0


with open(filename) as csv_file:
	csv_reader = csv.reader(csv_file)
	attributes=next(csv_reader)
	for row in csv_reader:
		if row[2] != '':
			countEmptyGross=countEmptyGross+float(row[2])
			total_count_gross=total_count_gross+1
		if row[7] != '':
			countEmptyIMDBScore=countEmptyIMDBScore+float(row[7])
			total_count_imdb=total_count_imdb+1

		if row[6] != '':
			countEmptyBudget=countEmptyBudget+float(row[6])
			total_count_budget=total_count_budget+1


meanGross=countEmptyGross/total_count_gross
meanIMDBScore=countEmptyIMDBScore/total_count_imdb		
meanBudget=countEmptyBudget/total_count_budget

dfOutcome=pd.DataFrame()
listOutcome=[]
netSpend=0
with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file)
    attributes = next(csv_reader)
    for row in csv_reader:
    	#print(row)
    	if row[2] != '' and row[7] != '':
    		data.append((float(row[2]),float(row[7])))
    	elif row[2] == '' and row[7] != '':
    		data.append((meanGross,float(row[7])))
    	elif row[2] == '' and row[7] != '':
    		data.append((float(row[7]),meanIMDBScore))
    	else:
    		data.append((meanGross,meanIMDBScore))
    	
    	if row[6] != '' and row[2] !='':
    		netSpend=float(row[2])-float(row[6])
    		if netSpend<0:
    			listOutcome.append('Unsuccessful')
    		else:
    			listOutcome.append('Successful')
    	elif row[6] == '' and row[2] != '':
    		netSpend=meanGross-float(row[2])
    		if netSpend<0:
    			listOutcome.append('Unsuccessful')
    		else:
    			listOutcome.append('Successful')
    	elif row[2] != '' and row[7] == '':
    		netSpend=float(row[2])-meanBudget
    		if netSpend<0:
    			listOutcome.append('Unsuccessful')
    		else:
    			listOutcome.append('Successful')
    	else:
    		netSpend=meanGross-meanBudget
    		if netSpend<0:
    			listOutcome.append('Unsuccessful')
    		else:
    			listOutcome.append('Successful')

print(listOutcome)




print("Datapoints extracted are: ",data)

k = int(input("Enter number of clusters required: "))
centroids=[]
i = 0
while i != k:
	point = tuple(data[random.randint(0,(len(data)-1))])
	if point not in centroids:
		centroids.append(point)
		i = i + 1
print("Randomly assigned centroids are: ",centroids)
old_clusters = formClusters(centroids,data)
print("First cluster: ",old_clusters)
counter = 0
new_clusters = old_clusters
while True:
	new_k_cent = []
	for i in range(k):
		k_th_clus = [x for x,y in new_clusters.items() if y == i]
		new_k_cent.append(compCentroid(k_th_clus,2))
	print("new centroids: ",new_k_cent)
	old_clusters = new_clusters
	new_clusters = formClusters(new_k_cent,data)
	print("new cluster: ",new_clusters)
	if new_clusters == old_clusters:
		counter = counter + 1
		if counter == 2:
			break
#print("Final cluster obtained is : ",new_clusters)
#print("Centroids are: ",new_k_cent)
#plot_cluster(new_clusters,new_k_cent)


#create new df 
dfOutcome = pd.DataFrame({'Net Earnings':listOutcome})


balance_data = pd.read_csv("movie.csv",sep= ',')

balance_data["director_name"].fillna( method ='bfill', inplace = True) 
balance_data["actor_1_name"].fillna( method ='bfill', inplace = True) 
balance_data["actor_2_name"].fillna( method ='bfill', inplace = True) 
balance_data["actor_3_name"].fillna( method ='bfill', inplace = True) 


#print(balance_data.head())

df1 =pd.DataFrame(balance_data)
df=pd.concat([df1, dfOutcome], axis=1)
#
##print(df.head())
#print(df.info())
#print("hello")
#print(df['director_name'].unique()[35])

#print(df.columns)
df['director_name'],_ = pd.factorize(df['director_name'])
df['actor_2_name'],_ = pd.factorize(df['actor_2_name'])
df['actor_1_name'],_ = pd.factorize(df['actor_1_name'])
df['actor_3_name'],_ = pd.factorize(df['actor_3_name'])
df['Net Earnings'],class_names=  pd.factorize(df['Net Earnings'])


X = df.values[:, [0,1,3,4]]
Y = df.values[:,8]
X=X.astype('float')
Y=Y.astype('float')
#print(X.shape)
#print(Y.dtype)

print("--Decision Tree--")
 #split data randomly into 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=0)
sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)  

dtree= DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=0)
dtree.fit(X_train, y_train)
##
## use the model to make predictions with the test data
y_pred = dtree.predict(X_test)
# how did our model perform?
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy through decision tree: {:.2f}'.format(accuracy))


feature_names = [df.columns[0],df.columns[1],df.columns[3],df.columns[4]]

dot_data = tree.export_graphviz(dtree, out_file=None, filled=True, rounded=True,
                                feature_names=feature_names,  
                                class_names=class_names)
#graph = graphviz.Source(dot_data)  
#graph.view()


#logistic regression code:

print("--Logistic Regression--")
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=0)
sc = StandardScaler()  
x_train = sc.fit_transform(x_train)  
x_test = sc.transform(x_test)  
print(y_train)

#using score method to get accuracy of the code
#accuracy (score method): correct predictions / total number of data points
logisticRegr = LogisticRegression()
##


#
logisticRegr.fit(x_train, y_train)
# Make predictions on entire test data
predictions = logisticRegr.predict(x_test)

score = logisticRegr.score(x_test, y_test)

print("Accuracy through logistic regression: ",score)


cm = metrics.confusion_matrix(y_test, predictions)


plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);
#plt.savefig('MovieAnalytics.png')
#plt.show();


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
scaler = StandardScaler()  

x_train = scaler.fit_transform(x_train)  
x_test = scaler.transform(x_test)  


classifier = KNeighborsClassifier(n_neighbors=4)  
classifier.fit(x_train, y_train)  

y_pred = classifier.predict(x_test)  

cm=metrics.confusion_matrix(y_test, y_pred) 
print("KNN confusion matrix:",cm) 
#
cr=metrics.classification_report(y_test, y_pred) 
print("KNN classification report: ",cr)
#
ac=accuracy_score(y_test, y_pred)
print(ac)
error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error') 
#plt.savefig('KNNAnalytics.png')
#plt.show();


#Random Forest Classification:

print("--Random Forest--")
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=0) 

sc = StandardScaler()  
x_train = sc.fit_transform(x_train)  
x_test = sc.transform(x_test)  

regressor = RandomForestRegressor(n_estimators=20, random_state=0)  
regressor.fit(x_train, y_train)  
y_pred = regressor.predict(x_test) 
y_pred=np.rint(y_pred)

cm=metrics.confusion_matrix(y_test,y_pred)  
print(cm)
cr=metrics.classification_report(y_test,y_pred)  
print(cr)
ac=accuracy_score(y_test, y_pred)
print("Accuracy through Random Forest: ",ac)

#predict imdb score:

X = df.values[:, [0,1,3,4]]
Y = df.values[:,7]
X=X.astype('float')
Y=Y.astype('float')


X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=0) 

sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test) 
regressor = RandomForestRegressor(n_estimators=20, random_state=0)  
regressor.fit(X_train, y_train)  
y_pred = regressor.predict(X_test)  

print('For IMDB Score Prediction: ')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:',math.floor(float(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))))
