import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
df = pd.read_excel("animal_disease_dataset.xlsx")
#df = df.drop('Disease', axis=1)
columns="Animal Age Temperature Symptom1 Symptom2 Symptom3".split()
#columns="Age Temperature Symptom1 Symptom2 Symptom3".split()

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

df = handle_non_numerical_data(df)
x=pd.DataFrame(df,columns=columns)
y=df.Disease
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model=DecisionTreeRegressor(max_depth=12)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
def compute_accuracy(Y_true, Y_pred):  
    correctly_predicted = 0  
    # iterating over every label and checking it with the true sample  
    for true_label, predicted in zip(Y_true, Y_pred):  
        if true_label == predicted:  
            correctly_predicted += 1  
    # computing the accuracy score  
    accuracy_score = correctly_predicted / len(Y_true)  
    return accuracy_score  
#accuracy = accuracy_score(y_test, y_pred)
accuracy = compute_accuracy(y_test, y_pred)
print(f'Model accuracy: {accuracy}')
