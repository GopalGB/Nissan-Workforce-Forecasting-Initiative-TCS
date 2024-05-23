import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Title of the Streamlit app
st.title('Nissan Operational Costs Prediction')

# Add a description about the application
st.markdown("""
This application predicts the operational costs based on various input features.
It is designed to help Nissan in forecasting and managing their operational expenses efficiently.
""")

# Add instructions for using the application
st.markdown("""
### Instructions:
1. Enter the values for each input feature.
2. Click the 'Predict' button to get the predicted operational cost.
3. Ensure all input values are realistic and within the expected range.
""")

# Input features
st.header('Input Features')

# Define all the features used in the model with descriptive labels and tooltips
monthly_production_volume = st.number_input('Monthly Production Volume', min_value=0, step=1, help='Total units produced per month')
number_of_employees = st.number_input('Number of Employees', min_value=0, step=1, help='Total number of employees involved in production')
average_equipment_downtime = st.number_input('Average Equipment Downtime (hours)', min_value=0.0, step=0.1, help='Average hours of equipment downtime per month')
raw_material_cost = st.number_input('Raw Material Cost', min_value=0.0, step=0.1, help='Total cost of raw materials per month')
machine_maintenance_cost = st.number_input('Machine Maintenance Cost', min_value=0.0, step=0.1, help='Total cost of machine maintenance per month')
logistics_cost = st.number_input('Logistics Cost', min_value=0.0, step=0.1, help='Total logistics cost per month')
energy_consumption_cost = st.number_input('Energy Consumption Cost', min_value=0.0, step=0.1, help='Total energy consumption cost per month')

# Collect all feature inputs in a list
features = [
    monthly_production_volume,
    number_of_employees,
    average_equipment_downtime,
    raw_material_cost,
    machine_maintenance_cost,
    logistics_cost,
    energy_consumption_cost
]

# Ensure the length of features list matches the number of features the model was trained on
if st.button('Predict'):
    # Convert features to a numpy array and reshape for a single sample
    input_features = np.array(features).reshape(1, -1)
    
    try:
        # Make prediction
        prediction = model.predict(input_features)
        # Display the prediction
        st.success(f'Predicted Operational Cost: {prediction[0]:.2f}')
    except ValueError as e:
        st.error(f"Error in prediction: {e}")

# Add a section about how Nissan can use this application
st.markdown("""
### How Nissan Can Use This Application:
Nissan can utilize this application to:
1. **Forecast Operational Costs**: By inputting the current production data, Nissan can predict future operational costs and plan budgets accordingly.
2. **Optimize Resource Allocation**: Understanding the impact of various factors such as equipment downtime and raw material costs helps in optimizing resource allocation.
3. **Improve Efficiency**: By regularly monitoring the predicted costs and comparing them with actual expenses, Nissan can identify areas for improvement and take corrective actions.
4. **Strategic Decision Making**: The application provides valuable insights that assist in making strategic decisions regarding production volumes, staffing, and maintenance schedules.
""")
