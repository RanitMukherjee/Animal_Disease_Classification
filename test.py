import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
predictions = model.predict(x_test)
print(predictions)
