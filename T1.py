
# AI's analytical power enables fund managers to make well-informed, data-driven decisions,
# predict market trends, assess risks, and optimize returns. This predictive capacity is poised to enhance
# the sustainability of micro-pensions substantially. Streetfins' report highlights the potential for AI to cut
# costs, customise portfolios, and identify market anomalies. However, it also acknowledges that AI might not
# predict market movements with complete accuracy, leaving an element of uncertainty.AI's analytical power
#  enables fund managers to make well-informed, data-driven decisions, predict market trends, assess risks,
#  and optimize returns. This predictive capacity is poised to enhance the sustainability of micro-pensions 
#  substantially. Streetfins' report highlights the potential for AI to cut costs, customise portfolios, and
#  identify market anomalies. However, it also acknowledges that AI might not predict market movements with 
#  complete accuracy, leaving an element of uncertainty.



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Sample dataset (for demonstration purposes)
data = {
    'Age': [25, 30, 35, 40, 45],
    'Income': [50000, 60000, 75000, 90000, 100000],
    'Savings': [20000, 30000, 40000, 50000, 60000],
    'Retirement_Savings_Goal': [300000, 400000, 500000, 600000, 700000]
}

df = pd.DataFrame(data)

# Features (Age, Income, Savings)
X = df[['Age', 'Income', 'Savings']]

# Target variable (Retirement Savings Goal)
y = df['Retirement_Savings_Goal']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor model
model = RandomForestRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the trained model for future use
joblib.dump(model, 'retirement_model.joblib')

# Use the model to predict retirement savings goal for a new user
new_user_data = pd.DataFrame({'Age': [28], 'Income': [55000], 'Savings': [25000]})
predicted_goal = model.predict(new_user_data)

print(f"Predicted Retirement Savings Goal: ${predicted_goal[0]:,.2f}")
