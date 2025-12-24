import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report

# Importing Dataset using Pandas
df = pd.read_csv("C:/Users/Bhawna/Downloads/archive (2)/heart_disease_dataset.csv")
#Filling null values
df['Alcohol Intake'].fillna(df['Alcohol Intake'].mode()[0],inplace = True)
# Data preprocessing , categorical to numeric
df = pd.get_dummies(df , columns = ['Alcohol Intake'], drop_first= True)
df['Diabetes'].replace({'Yes' : 1, 'No' : 0 },inplace  = True)
df['Obesity'].replace({'Yes' : 1, 'No' : 0 },inplace  = True)
df['Family History'].replace({'Yes' : 1, 'No' : 0 },inplace  = True)
df['Exercise Induced Angina'].replace({'Yes' : 1, 'No' : 0 },inplace  = True)
df = pd.get_dummies(df , columns = ['Smoking'], drop_first= True)
df['Gender'].replace({'Male' : 1, 'Female' : 0 },inplace  = True)
df = pd.get_dummies(df , columns = ['Chest Pain Type'], drop_first= True)

#removing target column
X = df.drop('Heart Disease' , axis = 1)
y = df['Heart Disease']
feature_names = X.columns # gives all the columns after doing preprocessing
# Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)
scaler = StandardScaler() # scaling
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(class_weight= 'balanced') # used logistic regression for prediction
model.fit(X_train_scaled , y_train)
y_pred = model.predict(X_test_scaled)
# for knowing how well the model works
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix" , cm)
print(classification_report(y_test, y_pred))

print("\n --- Enter Patient Details ---")
# Taking User input in categorical values
age = float(input("Enter Age: "))
gen = (input("Enter Gender: ")).lower()
chol = float(input("Enter Cholesterol:"))
bp = float(input("Enter Blood Pressure:"))
smoking = (input("Enter smoking (Yes/No):")).lower()
heart_rate = float(input("Enter Heart Rate:"))
alcohol_intake = (input("Enter Alcohol Intake(Heavy/None/Moderate):")).lower()
diabetes = (input("Enter Diabetes(yes/no):")).lower()
obesity = (input("Enter Obesity(yes/no):")).lower()
exercise_hour = float(input("Enter Exercise Hour:"))
stress_level = float(input("Enter Stress Level(1-10):"))
family_history = (input("Family History of heart disease:")).lower()
exercise_angina = (input("Do you have exercise Angina:")).lower()
chest_pain = (input("Enter Chest Pain Type:")).lower()

# creating a dataframe
user_df = pd.DataFrame(0, index=[0], columns=feature_names)
user_df['Age'] = age
user_df['Gender'] = gen
user_df['Cholesterol'] = chol
user_df['Blood Pressure'] = bp
user_df['Heart Rate'] = heart_rate
user_df['Exercise Hours'] = exercise_hour
user_df['Stress Level'] = stress_level

# matching categorical values and numeric
user_df['Diabetes'] = 1 if diabetes == 'yes' else 0
user_df['Obesity'] = 1 if obesity == 'yes' else 0
user_df['Family History'] = 1 if family_history == 'yes' else 0
user_df['Exercise Induced Angina'] = 1 if exercise_angina == 'yes' else 0
user_df['Gender'] =1 if gen == 'male' else 0

if smoking == 'yes' and 'Smoking_Yes' in user_df.columns:
    user_df['Smoking_Yes'] = 1
if alcohol_intake== 'heavy' and 'Alcohol Intake_Heavy' in user_df.columns:
    user_df['Alcohol Intake_Heavy'] = 1
elif alcohol_intake == 'moderate' and 'Alcohol Intake_Moderate' in user_df.columns:
    user_df['Alcohol Intake_Moderate'] = 1
if chest_pain == 'atypical' and 'Chest Pain Type_Atypical Angina' in user_df.columns:
    user_df['Chest Pain Type_Atypical Angina'] = 1
elif chest_pain == 'non-anginal' and 'Chest Pain Type_Non-anginal Pain' in user_df.columns:
    user_df['Chest Pain Type_Non-anginal Pain'] = 1
elif chest_pain == 'asymptomatic' and 'Chest Pain Type_Asymptomatic' in user_df.columns:
    user_df['Chest Pain Type_Asymptomatic'] = 1

# Scaling the new dataframe
user_input_scaled = scaler.transform(user_df)
prediction = model.predict(user_input_scaled)
# model gives probability
probability = model.predict_proba(user_input_scaled)

# Output
print("\n---Prediction Result---")
if prediction[0] == 1 :
    print("Heart Disease Detected")
else :
    print("No Heart Disease Detected")

print("\n---Probability Result---")
if probability[0][1] > 0.7 :
    print("High Risk")
elif probability[0][1] > 0.4:
    print("Medium Risk")
else :
    print("Low Risk")
# This is an Educational Model
