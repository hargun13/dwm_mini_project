import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import traceback
from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
from transformers import TextClassificationPipeline
import tensorflow as tf

model = joblib.load("C:/Users/HARGUN/Desktop/PROJECTS/DWM Mini Project/LinearRegressionModel.pkl")
tokenizer_fine_tuned = DistilBertTokenizer.from_pretrained(r"C:\Users\HARGUN\Desktop\PROJECTS\DWM Mini Project\saved_model")
model_fine_tuned = TFDistilBertForSequenceClassification.from_pretrained(r"C:\Users\HARGUN\Desktop\PROJECTS\DWM Mini Project\saved_model")

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        st.error("File not found. Please check the file path.")
        return None

def linear_regression(input_data):
    # Replace this with your actual model prediction logic
    predicted_price = model.predict(input_data)

    # Display the predicted price
    st.subheader(f"Predicted Car Price: ₹{predicted_price[0]:,.2f}")
    for _ in range(60):
        st.text("")

def data_visualization(data):
    st.subheader("Data Visualization:")
    st.write("Columns available for visualization:")
    st.write(data.columns)

    visualization_choice = st.selectbox("Select visualization type:", ["Histogram", "Box Plot", "Scatter Plot"])
    try:
        if visualization_choice == "Histogram":
            column = st.selectbox("Select column for histogram:", data.columns)
            fig, ax = plt.subplots()
            ax.hist(data[column])
            st.pyplot(fig)
            st.pyplot(plt.title(f'Histogram of {column}'))

        elif visualization_choice == "Box Plot":
            column = st.selectbox("Select column for box plot:", data.columns)

            # Check if the selected column can be converted to numeric
            data[column] = pd.to_numeric(data[column])

            fig, ax = plt.subplots()
            sns.boxplot(x=column, data=data, ax=ax)
            st.pyplot(fig)
            st.pyplot(plt.title(f'Box Plot of {column}'))

        elif visualization_choice == "Scatter Plot":
            x_column = st.selectbox("Select X-axis column:", data.columns)
            y_column = st.selectbox("Select Y-axis column:", data.columns)
            fig, ax = plt.subplots()
            ax.scatter(data[x_column], data[y_column])
            st.pyplot(fig)
            st.pyplot(plt.title(f'Scatter Plot of {x_column} vs {y_column}'))
            st.pyplot(plt.xlabel(x_column))
            st.pyplot(plt.ylabel(y_column))


    except Exception as e:
        print(f"Error in data visualization: {e}")
        traceback.print_exc()


def main():
    global uploaded_file
    global file_path
    st.title("Data Analysis with Streamlit")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    file_path = st.text_input("Enter the path of the CSV file:")
    data = None

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    elif file_path:
        data = load_data(file_path)
    else:
        st.warning("Please upload a file or enter the path of a CSV file.")

    if data is not None:
        display_menu()
        while True:
            operation_choice = st.selectbox("Select operation:", ["Show Data", "Descriptive Statistics", "Drop a Column", "Data Visualization", "Remove Missing Values", "Detect Outliers"] + (["ML Classifier"] if (file_path == r"C:\Users\HARGUN\Desktop\PROJECTS\DWM Mini Project\bbc-text.csv" or (uploaded_file is not None and uploaded_file.name == "bbc-text.csv")) else []) + (["ML Prediction"] if (file_path == r"C:\Users\HARGUN\Desktop\PROJECTS\DWM Mini Project\Cleaned_Car_data.csv" or (uploaded_file is not None and uploaded_file.name == "Cleaned_Car_data.csv")) else []) + ["Exit"])


            if operation_choice == "Show Data":
                st.subheader("Data:")
                st.write(data)
                for _ in range(60):
                    st.text("")

            elif operation_choice == "Descriptive Statistics":
                st.subheader("Descriptive Statistics:")
                st.write(data.describe())
                for _ in range(60):
                    st.text("")
            
            
            elif operation_choice == "Drop a Column":
                st.subheader("Drop Column:")
                st.write("Current columns:")
                st.write(data.columns)

                column_to_drop = st.text_input("Enter the column name to drop:")
                if st.button("Drop Column"):
                    if column_to_drop in data.columns:
                        data.drop(column_to_drop, axis=1, inplace=True)
                        st.success(f"Column '{column_to_drop}' dropped successfully.")
                        st.write("Updated dataset")
                        st.write(data.columns)
                    else:
                        st.error(f"Column '{column_to_drop}' not found.")
                        for _ in range(60):
                            st.text("")
                for _ in range(60):
                        st.text("")
            
            
            
            elif operation_choice == "Data Visualization":
                data_visualization(data)
                for _ in range(60):
                    st.text("")

            elif operation_choice == "ML Prediction" and (file_path == r"C:\Users\HARGUN\Desktop\PROJECTS\DWM Mini Project\Cleaned_Car_data.csv" or uploaded_file.name) == "Cleaned_Car_data.csv":
                perform_ml_prediction(data)

            elif operation_choice == "Remove Missing Values":
                data = remove_missing_values(data)
                st.success("Missing values removed successfully.")
                st.write(data)
                for _ in range(60):
                    st.text("")
            
            elif operation_choice == "Detect Outliers":
                outliers = detect_outliers(data)
                st.subheader("Outliers:")
                st.write(outliers)
                for _ in range(60):
                    st.text("")
            
            elif operation_choice == "ML Classifier" and (file_path == r"C:\Users\HARGUN\Desktop\PROJECTS\DWM Mini Project\bbc-text.csv" or uploaded_file.name) == "bbc-text.csv":
                prediction_value = classification()
                st.subheader("Class:")
                if prediction_value == 0:
                    st.write("Belongs to the Business class")
                elif prediction_value == 1:
                    st.write("Belongs to the Entertainment  class")
                elif prediction_value == 2:
                    st.write("Belongs to the Politics  class")
                elif prediction_value == 3:
                    st.write("Belongs to the Sports class")
                elif prediction_value == 4:
                    st.write("Belongs to the Technology class")
                else:
                    st.write("None")
                for _ in range(60):
                    st.text("")

            elif operation_choice == "Exit":
                st.write("Exiting the program. Goodbye!")
                break

def remove_missing_values(data):
    # Iterate through columns
    for column in data.columns:
        # Check if the column has missing values
        if data[column].isnull().any():
            # Check data type of the column
            if pd.api.types.is_numeric_dtype(data[column]):
                # If numeric, fill missing values with the mean
                data[column].fillna(data[column].mean(), inplace=True)
            else:
                # If text, fill missing values with the most occurring value (mode)
                mode_value = data[column].mode().iloc[0]
                data[column].fillna(mode_value, inplace=True)

    return data


def classification():
    test_text = st.text_area("Enter the text:")


    predict_input = tokenizer_fine_tuned.encode(
        test_text,
        truncation = True,
        padding = True,
        return_tensors = 'tf'
    )

    output = model_fine_tuned(predict_input)[0]

    prediction_value = tf.argmax(output, axis = 1).numpy()[0]
    st.write(prediction_value)

    return prediction_value


def detect_outliers(data):
    outliers = pd.DataFrame()
    
    # Iterate through columns
    for column in data.select_dtypes(include='number').columns:
        # Calculate the IQR for the column
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1

        # Define the lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Detect outliers and add to the outliers DataFrame
        column_outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        outliers = pd.concat([outliers, column_outliers])

    st.success("Following Outliers were detected")

    return outliers


def perform_ml_prediction(data):
    st.subheader("ML Prediction:")
    
    # Select fields for ML prediction
    company = st.selectbox("Select Company:", data['company'].unique())
    model_name = st.selectbox("Select Model:", data[data['company'] == company]['name'].unique())
    year = st.selectbox("Select Year:", data[data['name'] == model_name]['year'].unique())
    fuel_type = st.selectbox("Select Fuel Type:", data[data['name'] == model_name]['fuel_type'].unique())
    kms_driven = st.number_input("Enter Kilometers Driven:", min_value=0)

    # Prepare input data for prediction
    input_data = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                               data=[[model_name, company, year, kms_driven, fuel_type]])

    # Perform ML prediction
    linear_regression(input_data)

def display_menu():
    st.sidebar.subheader("Menu:")
    st.sidebar.write("1. Show Data")
    st.sidebar.write("2. Descriptive Statistics")
    st.sidebar.write("3. Drop a Column")
    st.sidebar.write("4. Data Visualization")
    if file_path == r"C:\Users\HARGUN\Desktop\PROJECTS\DWM Mini Project\Cleaned_Car_data.csv" or uploaded_file.name == "Cleaned_Car_data.csv":
        st.sidebar.write("5. ML Prediction")
    else:
        st.sidebar.markdown("- ~~5. ML Prediction~~")
    st.sidebar.write("6. Remove Missing Values")
    st.sidebar.write("7. Detect Outliers")
    if file_path == r"C:\Users\HARGUN\Desktop\PROJECTS\DWM Mini Project\bbc-text.csv" or uploaded_file.name == "bbc-text.csv":
        st.sidebar.write("8. ML Classifier")
    else:
        st.sidebar.markdown("- ~~8. ML Classifier~~")
    st.sidebar.write("9. Exit")

if __name__ == "__main__":
    main()
