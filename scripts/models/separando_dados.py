#%%
import pandas as pd 
from pre_processing import separar_base

#%%
data = pd.read_csv('..\\..\\data\\processed\\final_data.csv') 
X_train, X_val, X_test, y_train, y_val, y_test = separar_base(data)



# %%
X_train.to_csv("Train&TestData\\X_train.csv")
X_val.to_csv("Train&TestData\\X_val.csv")
X_test.to_csv("Train&TestData\\X_test.csv")
y_train.to_csv("Train&TestData\\y_train.csv")
y_val.to_csv("Train&TestData\\y_val.csv")
y_test.to_csv("Train&TestData\\y_test.csv")
# %%
