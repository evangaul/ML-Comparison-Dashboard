import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
import warnings

# Ignore warnings for better output
warnings.filterwarnings("ignore")

st.title("Interactive Machine Learning Model Comparison Dashboard")

# This is the sidebar, will have header, place to upload file, model selection, test size selection
st.sidebar.header("Dataset and Model Parameters")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
model_choice = st.sidebar.multiselect("Select Models",
    ["Logistic Regression", "Random Forest", "Support Vector Machine"],
    default=["Logistic Regression", "Random Forest"])
test_size = st.sidebar.slider("Test Set Size (%)", 10, 50, 20) / 100
encode_categorical = st.sidebar.checkbox("Encode Categorical Columns", value=True)
scale_features = st.sidebar.checkbox("Scale Features (Recommended for SVM)", value=True)

# Initialize session state for results
if "results" not in st.session_state:
    st.session_state.results = {}

# Process the uploaded file
if uploaded_file:
    try:
        # Load dataset
        df = pd.read_csv(uploaded_file)
        st.write("**Data Preview**")
        rows, columns = df.shape
        st.write("Number of columns: " + str(columns))
        st.write("Number of rows: " + str(rows))

        st.dataframe(df.head()) # Display df head

        # Select target column
        columns = df.columns.tolist()
        target_column = st.sidebar.selectbox("Select Target Column", columns)

        # Validate the target column
        if target_column:
            # target has to be the correct type
            if not df[target_column].dtype in [np.object_, np.int64, np.float64, np.bool_]:
                st.error("Target column must be categorical or numeric for classification.")
            else:
                X = df.drop(columns=[target_column]) # target column
                y = df[target_column] # Features

                # Handle categorical columns, if encode checkbox selected
                if encode_categorical:
                    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
                    if len(categorical_cols) > 0:
                        # Limit encoding to columns with reasonable unique values
                        valid_categorical = [col for col in categorical_cols if X[col].nunique() < 20]
                        if len(valid_categorical) < len(categorical_cols):
                            st.warning(f"Skipping high-cardinality columns: {set(categorical_cols) - set(valid_categorical)}")
                        X = pd.get_dummies(X, columns=valid_categorical, drop_first=True)
                    else:
                        st.info("No categorical columns detected for encoding.")

                # Handle missing values
                if X.isnull().any().any() or y.isnull().any():
                    st.warning("Missing values found, Dropping rows with missing values!")
                    data = pd.concat([X, y], axis=1).dropna()
                    X = data.drop(columns=[target_column])
                    y = data[target_column]

                # Make sure size of dataset is good
                if len(X) < 10:
                    st.error("Dataset too small after preprocessing.")
                elif len(X.columns) < 1:
                    st.error("No features available after preprocessing.")
                else:
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                    # Scale features if selected
                    if scale_features:
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)

                    # Train and evaluate models
                    st.header("Model Comparison")
                    results = {}
                    for model_name in model_choice:
                        if model_name == "Logistic Regression":
                            model = LogisticRegression(max_iter=1000)
                        elif model_name == "Random Forest":
                            model = RandomForestClassifier(random_state=42)
                        elif model_name == "Support Vector Machine":
                            model = SVC(random_state=42)

                        # Train model
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                        # Compute accuracy and precision
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        results[model_name] = {"accuracy": accuracy, "precision": precision, "model": model}

                        # Write accuracy and precision
                        st.subheader(model_name)
                        st.write(f"Accuracy: {accuracy:.2%}")
                        st.write(f"Precision: {precision:.2%}")

                    # Store results in session state
                    st.session_state.results = results

                    # Plot confusion matrix for each model
                    st.header("Confusion Matrices")
                    for model_name, result in results.items():
                        cm = confusion_matrix(y_test, result["model"].predict(X_test))
                        fig = px.imshow(
                            cm,
                            labels=dict(x="Predicted", y="Actual", color="Count"),
                            title=f"Confusion Matrix ({model_name})"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Plot feature importance for Random Forest
                    if "Random Forest" in results and not scale_features:
                        importances = results["Random Forest"]["model"].feature_importances_
                        fig = px.bar(
                            x=X.columns,
                            y=importances,
                            title="Feature Importance (Random Forest)",
                            labels={"x": "Feature", "y": "Importance"}
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Compare model performance
                    if len(results) > 1:
                        st.header("Model Performance Comparison")
                        metrics_df = pd.DataFrame({
                            "Model": results.keys(),
                            "Accuracy": [results[m]["accuracy"] for m in results],
                            "Precision": [results[m]["precision"] for m in results]
                        })
                        fig = px.bar(
                            metrics_df,
                            x="Model",
                            y=["Accuracy", "Precision"],
                            barmode="group",
                            title="Model Performance Comparison"
                        )
                        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing the dataset: {e}")

else:
    st.info("Please upload a CSV file to begin.")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Developed by Evan Gaul")
st.sidebar.write("Github: https://github.com/evangaul/ML-Comparison-Dashboard.git")
