from sklearn import tree

# [height, weight, shoe size] for 11 people

x = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37],
     [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40],
     [159, 55, 37], [171, 75, 42], [181, 85, 43]]

#  list of labels associated with the body metrics in the variable x

y = ['male', 'female', 'female', 'female', 'male', 'male',
     'male', 'female', 'male', 'female', 'male']

#  decision tree classifier

clf = tree.DecisionTreeClassifier()

# .fit trains the decisiontree on the dataset that has been given (x, y)

clf = clf.fit(x, y)

#  predict the gender of new data

prediction = clf.predict([[190, 70, 43]])

print(prediction)
