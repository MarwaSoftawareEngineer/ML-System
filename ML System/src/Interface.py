import os
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# For Data Visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu


from PreparingModel import (read_data,
                        preprocess_data,
                        train_model,
                        evaluate_model)


# Get the working directory of the main.py file
working_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(working_dir)


st.set_page_config(
    page_title="ML Model Training",
    page_icon="🧠",
    layout="centered")

# sidebar for navigation
with st.sidebar:
    selected = option_menu('ML System',

                           ['ML Model Training',
                            'Data Visualizer'],
                           menu_icon='robot',
                           icons=['robot', 'tv'],
                           default_index=0)
    
# ML Model Training Page
if selected == 'ML Model Training':
    # page title
    st.title('🤖 ML Model Training')    



    dataset_list = os.listdir(f"{parent_dir}/data")

    dataset = st.selectbox("Select a dataset from the dropdown",
                        dataset_list,
                        index=None)

    df = read_data(dataset)

    if df is not None:
        st.dataframe(df)

        col1, col2, col3, col4 = st.columns(4)

        scaler_type_list = ["standard", "minmax"]

        model_dictionary = {
            "Logistic Regression": LogisticRegression(),           # Instantiate Logistic Regression model
            "Support Vector Classifier": SVC(),                    # Instantiate Support Vector Classifier model
            "Random Forest Classifier": RandomForestClassifier(),  # Instantiate Random Forest Classifier model
            "KNN Classifier": KNeighborsClassifier()               # Instantiate KNN Classifier model
        }


        with col1:
            target_column = st.selectbox("Select the Target Column", list(df.columns))
        with col2:
            scaler_type = st.selectbox("Select a scaler", scaler_type_list)
        with col3:
            selected_model = st.selectbox("Select a Model", list(model_dictionary.keys()))
        with col4:
            model_name = st.text_input("Model name")

        if st.button("Train the Model"):

            X_train, X_test, y_train, y_test = preprocess_data(df, target_column, scaler_type)

            model_to_be_trained = model_dictionary[selected_model]

            model = train_model(X_train, y_train, model_to_be_trained, model_name)

            accuracy = evaluate_model(model, X_test, y_test)

            st.title("Your Model is: " + str(model_name))
            

            st.success("Test Accuracy: " + str(accuracy))


# Data Visualizer Page
if selected == "Data Visualizer":
    # page title
    st.title("📊 Data Visualizer")

    working_dir = os.path.dirname(os.path.abspath(__file__))

    # Specify the folder where your CSV files are located
    folder_path = os.path.join(working_dir, "data")  # Update this to your folder path

    # Check if the folder exists
    if not os.path.exists(folder_path):
        st.error(f"The folder '{folder_path}' does not exist.")
    else:
        # List all files in the folder
        files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

        if not files:
            st.error(f"No CSV files found in '{folder_path}'")
        else:
            # Dropdown to select a file
            selected_file = st.selectbox("Select a file", files, index=None)

            if selected_file:
                # Construct the full path to the file
                file_path = os.path.join(folder_path, selected_file)

                # Read the selected CSV file
                df = pd.read_csv(file_path)

                col1, col2 = st.columns(2)

                columns = df.columns.tolist()

                with col1:
                    st.write("")
                    st.write(df.head())

                with col2:
                    # Allow the user to select columns for plotting
                    x_axis = st.selectbox("Select the X-axis", options=columns + ["None"])
                    y_axis = st.selectbox("Select the Y-axis", options=columns + ["None"])

                    plot_list = [
                        "Line Plot",
                        "Bar Chart",
                        "Scatter Plot",
                        "Distribution Plot",
                        "Count Plot",
                        "Pie Chart",
                        "Histogram",
                        "Box Plot",
                        "Heatmap",
                        "Violin Plot",
                        "Joint Plot"
                        ]

                    # Allow the user to select the type of plot
                    plot_type = st.selectbox("Select the type of plot", options=plot_list)

                # Generate the plot based on user selection
                if st.button("Generate Plot"):
                    fig, ax = plt.subplots(figsize=(6, 4))
                    # Create a Line Plot
                    if plot_type == "Line Plot":
                        sns.lineplot(x=df[x_axis], y=df[y_axis], ax=ax)
                    # Create a Bar Chart
                    elif plot_type == "Bar Chart":
                        sns.barplot(x=df[x_axis], y=df[y_axis], ax=ax)
                    # Create a Scatter Plot
                    elif plot_type == "Scatter Plot":
                        sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax)
                    # Create a Distribution Plot
                    elif plot_type == "Distribution Plot":
                        sns.histplot(df[x_axis], kde=True, ax=ax)
                        y_axis = "Density"
                    # Create a Bar Chart
                    elif plot_type == "Count Plot":
                        sns.countplot(x=df[x_axis], ax=ax)
                        y_axis = "Count"
                    # Create a Bar Chart
                    elif plot_type == "Pie Chart":
                    # Count the frequency of each category in a column
                        category_counts = df[x_axis].value_counts()
                    # Create a pie chart
                        ax.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
                        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

                    elif plot_type == "Histogram":
                    # Create a histogram
                        sns.histplot(df[x_axis], kde=False, ax=ax)
                        ax.set_ylabel("Frequency")  # Set y-axis label

                    elif plot_type == "Box Plot":
                    # Create a box plot
                        sns.boxplot(x=df[x_axis], ax=ax)

                    elif plot_type == "Heatmap":
                    # Create a heatmap of correlations
                        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)

                    elif plot_type == "Violin Plot":
                    # Create a violin plot
                        sns.violinplot(x=df[x_axis], y=df[y_axis], ax=ax)

                    elif plot_type == "Joint Plot":
                    # Create a joint plot
                        sns.jointplot(x=df[x_axis], y=df[y_axis], kind='scatter')


                    # Adjust label sizes
                    ax.tick_params(axis="x", labelsize=10)  # Adjust x-axis label size
                    ax.tick_params(axis="y", labelsize=10)  # Adjust y-axis label size

                    # Adjust title and axis labels with a smaller font siz
                    plt.title(f"{plot_type} of {y_axis} vs {x_axis}", fontsize=12)
                    plt.xlabel(x_axis, fontsize=10)
                    plt.ylabel(y_axis, fontsize=10)

                    # Show the results
                    st.pyplot(fig)
