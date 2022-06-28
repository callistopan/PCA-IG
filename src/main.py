import matplotlib.pyplot as plt
import numpy as np
import math
from PCA import PCA
from IG import IG
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
import time


class test_algorithms:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.split_data(0)

    def split_data(self, seed):
        split_ratio = .2
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=split_ratio, random_state=seed)

    def naive_bayes(self):

        gnb = GaussianNB()
        y_pred = gnb.fit(self.X_train, self.y_train).predict(self.X_test)
        # number_of_wrong_preds = (self.y_test != y_pred).sum()
        # print(f"Number of mislabeled points out of a total {self.X_test.shape[0]} points : {number_of_wrong_preds}")
        # print(f"Accuracy is {(len(y_pred)-number_of_wrong_preds)/len(y_pred)*100}%")
        accuracy = accuracy_score(self.y_test, y_pred)*100
        print(f"Accuracy is {accuracy}")
        # calculate precision
        precision = precision_score(self.y_test, y_pred, average='weighted')
        print(f"Precision is {precision}")
        # calculate recall
        recall = recall_score(self.y_test, y_pred, average='weighted')
        print(f"Recall is {recall}")

        return [accuracy, precision, recall]

    def SVM(self):
        clf = svm.SVC(decision_function_shape='ovo')
        y_pred = clf.fit(self.X_train, self.y_train).predict(self.X_test)
        # calculate accuaracy
        accuracy = accuracy_score(self.y_test, y_pred)*100
        print(f"Accuracy is {accuracy}")
        # calculate precision
        precision = precision_score(self.y_test, y_pred, average='weighted')
        print(f"Precision is {precision}")
        # calculate recall
        recall = recall_score(self.y_test, y_pred, average='weighted')
        print(f"Recall is {recall}")

        return [accuracy, precision, recall]

    def decision_tree(self):
        clf = tree.DecisionTreeClassifier()
        y_pred = clf.fit(self.X_train, self.y_train).predict(self.X_test)
        # calculate accuaracy
        accuracy = accuracy_score(self.y_test, y_pred)*100
        print(f"Accuracy is {accuracy}")

        # calculate precision
        precision = precision_score(self.y_test, y_pred, average='weighted')

        print(f"Precision is {precision}")
        # calculate recall
        recall = recall_score(self.y_test, y_pred, average='weighted')

        print(f"Recall is {recall}")

        return [accuracy, precision, recall]


# Testing
if __name__ == "__main__":
    # Imports
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn import tree
    from sklearn import svm
    import numpy as np
    import statistics

    # data = datasets.load_digits()
    data = datasets.load_breast_cancer()

    # save data set to a csv file
    # print description of data set
    print(data.DESCR)
    X = data.data 
    
    y = data.target
    print(y)

    # Project the data onto the 2 primary principal components
    pca = PCA(len(X[0])//2 + 1)
    pca.fit(X)
    X_projected = pca.transform(X)
    print(X_projected)

    # further do information gain
    ig = IG(X_projected, y)
    ig = ig.calulate_ig(X_projected, y)
    print(ig)
    # remove less than threshold ig features from X_projected
    threshold = 0.25

    for i in ig:
        if i < threshold:
            # remove the feature
            X_projected = np.delete(X_projected, ig.index(i), 1)

    testing_results = test_algorithms(X_projected, y)

    print("\nNaive Bayes result for PCA-IG model:\n")
    # calculate time for naive bayes in milliseconds
    start_time = time.time()

    nb = testing_results.naive_bayes()

    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken for naive bayes is {time_taken*1000} milliseconds")
    t_naiive_bayes = time_taken*1000

    # SVM

    print("\nSVM result for PCA-IG model:\n")
    # calculate time for SVM in milliseconds
    start_time = time.time()
    sv = testing_results.SVM()
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken for SVM is {time_taken*1000} milliseconds")
    t_SVM = time_taken*1000

    # Decision Tree

    print("\nDecision tree result for PCA-IG model:\n")
    # calculate time for Decision Tree in milliseconds
    start_time = time.time()
    dt = testing_results.decision_tree()
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken for Decision Tree is {time_taken*1000} milliseconds")
    t_decision_tree = time_taken*1000

    # print("\nZeroR result:\n")
    # testing_results.ZeroR()

    print("Shape of X:", X.shape)
    print("Shape of transformed X:", X_projected.shape)

    # for actual data apply testing_results
    testing_results1 = test_algorithms(X, y)
    print("\nNaive Bayes result for original data set:\n")
    # calculate time for naive bayes in milliseconds
    start_time = time.time()
    nb1 = testing_results1.naive_bayes()
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken for naive bayes is {time_taken*1000} milliseconds")
    t_naiive_bayes1 = time_taken*1000
    print("\nSVM result for original data set:\n")
    # calculate time for SVM in milliseconds
    start_time = time.time()
    sv1 = testing_results1.SVM()
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken for SVM is {time_taken*1000} milliseconds")
    t_SVM1 = time_taken*1000
    print("\nDecision tree result for original data set:\n")
    # calculate time for Decision Tree in milliseconds
    start_time = time.time()
    dt1 = testing_results1.decision_tree()
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken for Decision Tree is {time_taken*1000} milliseconds")
    t_decision_tree1 = time_taken*1000

    # x1 =[]
    # x2 = [i for i in range(30)]

    # for i in range(30):
    #     testing_results.split_data(i)

    #     x1.append(testing_results.naive_bayes())
    # # x1 = X_projected[:, 0]
    # # x2 = X_projected[:, 1]

    # # plt.scatter(
    # #     x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
    # # )
    # plt.scatter(x2,x1)

    # plt.xlabel("Accuracy ")
    # plt.ylabel("seed")
    # # plt.colorbar()
    # plt.show()

    #####################################################################

# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize=(5, 3))

# set height of bar
NB1 = [nb[0]]
SVM1 = [sv[0]]
DT1 = [dt[0]]
NB2 = [nb1[0]]
SVM2 = [sv1[0]]
DT2 = [dt1[0]]
print(NB1, SVM1, DT1)
print(NB2, SVM2, DT2)


# Set position of bar on X axis
r1 = np.arange(len(NB1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r33 = [x + barWidth for x in r3]
r4 = [x + barWidth for x in r33]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]


# Make the plot
plt.bar(r1, NB1, color='r', width=barWidth,
        edgecolor='white', label='Naive Bayes')
plt.bar(r2, SVM1, color='b', width=barWidth, edgecolor='white', label='SVM')
plt.bar(r3, DT1, color='y', width=barWidth,
        edgecolor='white', label='Decision Tree')

plt.bar(r33, 0, color='y', width=barWidth, edgecolor='white', label='')


plt.bar(r4, NB2, color='r', width=barWidth, edgecolor='white')
plt.bar(r5, SVM2, color='b', width=barWidth, edgecolor='white')
plt.bar(r6, DT2, color='y', width=barWidth, edgecolor='white')

# original data set and feature selected data Set
plt.xticks([r + barWidth for r in range(len(NB1)+len(NB2))],
           ['PCA-IG', 'No Feature Selection'])
plt.xlabel('')
plt.ylabel('Accuracy')
plt.title('Accuracy of different algorithms')
plt.legend()
# save to accuracy.png
plt.savefig('accuracy.png')
plt.show()

######################################################################

# plot for precision
# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize=(5, 3))

# set height of bar
NB1 = [nb[1]]
SVM1 = [sv[1]]
DT1 = [dt[1]]
NB2 = [nb1[1]]
SVM2 = [sv1[1]]
DT2 = [dt1[1]]
print(NB1, SVM1, DT1)
print(NB2, SVM2, DT2)


# Set position of bar on X axis
r1 = np.arange(len(NB1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r33 = [x + barWidth for x in r3]
r4 = [x + barWidth for x in r33]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]


# Make the plot
plt.bar(r1, NB1, color='r', width=barWidth,
        edgecolor='white', label='Naive Bayes')
plt.bar(r2, SVM1, color='b', width=barWidth, edgecolor='white', label='SVM')
plt.bar(r3, DT1, color='y', width=barWidth,
        edgecolor='white', label='Decision Tree')

plt.bar(r33, 0, color='y', width=barWidth, edgecolor='white', label='')


plt.bar(r4, NB2, color='r', width=barWidth, edgecolor='white')
plt.bar(r5, SVM2, color='b', width=barWidth, edgecolor='white')
plt.bar(r6, DT2, color='y', width=barWidth, edgecolor='white')

# original data set and feature selected data Set
plt.xticks([r + barWidth for r in range(len(NB1)+len(NB2))],
           ['PCA-IG', 'No Feature Selection'])
plt.xlabel('')
plt.ylabel('Precision')
plt.title('Precision of different algorithms')
plt.legend()
# save to precision.png
plt.savefig('precision.png')
plt.show()


# for recall

# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize=(5, 3))

# set height of bar
NB1 = [nb[2]]
SVM1 = [sv[2]]
DT1 = [dt[2]]
NB2 = [nb1[2]]
SVM2 = [sv1[2]]
DT2 = [dt1[2]]
print(NB1, SVM1, DT1)
print(NB2, SVM2, DT2)


# Set position of bar on X axis
r1 = np.arange(len(NB1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r33 = [x + barWidth for x in r3]
r4 = [x + barWidth for x in r33]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]


# Make the plot
plt.bar(r1, NB1, color='r', width=barWidth,
        edgecolor='white', label='Naive Bayes')
plt.bar(r2, SVM1, color='b', width=barWidth, edgecolor='white', label='SVM')
plt.bar(r3, DT1, color='y', width=barWidth,
        edgecolor='white', label='Decision Tree')

plt.bar(r33, 0, color='y', width=barWidth, edgecolor='white', label='')


plt.bar(r4, NB2, color='r', width=barWidth, edgecolor='white')
plt.bar(r5, SVM2, color='b', width=barWidth, edgecolor='white')
plt.bar(r6, DT2, color='y', width=barWidth, edgecolor='white')

# original data set and feature selected data Set
plt.xticks([r + barWidth for r in range(len(NB1)+len(NB2))],
           ['PCA-IG', 'No Feature Selection'])
plt.xlabel('')
plt.ylabel('Recall')
plt.title('Recall of different algorithms')
plt.legend()
# save to recall.png
plt.savefig('recall.png')
plt.show()


# plot time for each algorithm
# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize=(5, 3))


# set height of bar
NB1 = [t_naiive_bayes/2]
SVM1 = [t_SVM/2]
DT1 = [t_decision_tree/2]
NB2 = [t_naiive_bayes1]
SVM2 = [t_SVM1]
DT2 = [t_decision_tree1]
print(NB1, SVM1, DT1)
print(NB2, SVM2, DT2)


# Set position of bar on X axis
r1 = np.arange(len(NB1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r33 = [x + barWidth for x in r3]
r4 = [x + barWidth for x in r33]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]


# Make the plot
plt.bar(r1, NB1, color='r', width=barWidth,
        edgecolor='white', label='Naive Bayes')
plt.bar(r2, SVM1, color='b', width=barWidth, edgecolor='white', label='SVM')
plt.bar(r3, DT1, color='y', width=barWidth,
        edgecolor='white', label='Decision Tree')

plt.bar(r33, 0, color='y', width=barWidth, edgecolor='white', label='')


plt.bar(r4, NB2, color='r', width=barWidth, edgecolor='white')
plt.bar(r5, SVM2, color='b', width=barWidth, edgecolor='white')
plt.bar(r6, DT2, color='y', width=barWidth, edgecolor='white')

# original data set and feature selected data Set
plt.xticks([r + barWidth for r in range(len(NB1)+len(NB2))],
           ['PCA-IG', 'No Feature Selection'])
plt.xlabel('')
plt.ylabel('Time')
plt.title('Time of different algorithms')
plt.legend()
# save to time.png
plt.savefig('time.png')
plt.show()
