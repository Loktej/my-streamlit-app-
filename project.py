import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import run_inference


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.title("Grocery Sales Prediction")


item_identifiers = ['FDA15', 'DRC01', 'FDN15', 'FDX07', 'NCD19']

# Dropdown options
dropdown_options = {
    "Item_Fat_Content": ["Low Fat", "Regular"],
    "Item_Type": [
        "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables",
        "Household", "Baking Goods", "Snack Foods", "Frozen Foods",
        "Breakfast", "Health and Hygiene", "Hard Drinks", "Canned",
        "Breads", "Starchy Foods", "Others", "Seafood"
    ],
    "Outlet_Identifier": [
        "OUT049", "OUT018", "OUT010", "OUT013", "OUT027",
        "OUT045", "OUT017", "OUT046", "OUT035", "OUT019"
    ],
    "Outlet_Size": ["Small", "Medium", "High"],
    "Outlet_Location_Type": ["Tier 1", "Tier 2", "Tier 3"],
    "Outlet_Type": [
        "Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"
    ],
}

# Value ranges for numeric inputs
value_ranges = {
    "Item_Visibility": (0.0, 0.328391),
    "Item_Weight": (4.555, 21.35),
    "Item_MRP": (31.29, 266.8884),
}

# Collect user inputs
user_inputs = {}

user_inputs["Item_Identifier"] = st.selectbox(
    "Select Item Identifier:", options=item_identifiers
)

for key, options in dropdown_options.items():
    user_inputs[key] = st.selectbox(f"Select {key}:", options)

user_inputs["Item_Weight"] = st.number_input(
    "Enter Item Weight:",
    min_value=value_ranges["Item_Weight"][0],
    max_value=value_ranges["Item_Weight"][1],
    value=(value_ranges["Item_Weight"][0] + value_ranges["Item_Weight"][1]) / 2,
    step=0.01
)

user_inputs["Item_MRP"] = st.number_input(
    "Enter Item MRP:",
    min_value=value_ranges["Item_MRP"][0],
    max_value=value_ranges["Item_MRP"][1],
    value=(value_ranges["Item_MRP"][0] + value_ranges["Item_MRP"][1]) / 2,
    step=0.01
)

user_inputs["Item_Visibility"] = st.number_input(
    "Enter Item Visibility:",
    min_value=value_ranges["Item_Visibility"][0],
    max_value=value_ranges["Item_Visibility"][1],
    value=(value_ranges["Item_Visibility"][0] + value_ranges["Item_Visibility"][1]) / 2,
    step=0.0001
)

years = list(range(1985, 2010))
user_inputs["Outlet_Establishment_Year"] = st.selectbox(
    "Select Outlet Establishment Year:", options=years
)


if st.button("Predict"):
    user_input_df = pd.DataFrame([user_inputs])
    try:
        predictions = run_inference(user_input_df)
        predicted_sales = predictions[0]
        st.success(f"Predicted Item Outlet Sales: ${predicted_sales:.2f}")

        
        st.subheader("Input Distribution")
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        sns.histplot(user_input_df["Item_Weight"], kde=True, ax=ax[0])
        ax[0].set_title("Item Weight Distribution")

        sns.histplot(user_input_df["Item_MRP"], kde=True, ax=ax[1])
        ax[1].set_title("Item MRP Distribution")

        sns.histplot(user_input_df["Item_Visibility"], kde=True, ax=ax[2])
        ax[2].set_title("Item Visibility Distribution")

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
