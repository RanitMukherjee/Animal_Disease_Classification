import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

#from six import StringIO
#from IPython.display import Image  
#from sklearn.tree import export_graphviz
#import pydotplus

col_names = ['Animal', 'Age', 'Temparature', 'Symptom1', 'Symptom2', 'Symptom3', 'Disease']
pima = pd.read_excel("animal_disease_dataset.xlsx", header=None, names=col_names)


def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

pima = handle_non_numerical_data(pima)
pima.head()
feature_cols = ['Age', 'Temparature', 'Symptom1', 'Symptom2', 'Symptom3']
X = pima[feature_cols] 
y = pima.Disease
#columns="Age Temperature Symptom1 Symptom2 Symptom3".split()
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


#dot_data = StringIO()
#export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True,feature_names = feature_cols,class_names=['0','1'])
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#graph.write_png('diabetes.png')
#Image(graph.create_png())
