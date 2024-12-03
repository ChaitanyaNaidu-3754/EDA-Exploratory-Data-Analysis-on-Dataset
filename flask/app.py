import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

import matplotlib
matplotlib.use('Agg')

# Initialize Flask app
app = Flask(__name__)

# Folder to save uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions for uploaded files
ALLOWED_EXTENSIONS = {'csv'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to load Seaborn dataset
def load_seaborn_dataset(dataset_name):
    try:
        df = sns.load_dataset(dataset_name)
        return df
    except Exception as e:
        return str(e)

# Function to process and analyze the dataset
def preprocess_data(df):
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].median())


    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode())
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

def perform_eda(df):
    # Saving the distribution and correlation plot images
    images = []
    
    # Distribution plots for numerical columns
    for col in df.select_dtypes(include='number').columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f'Distribution of {col}')
        image_path = os.path.join('static/images', f'{col}_distribution.png')
        plt.savefig(image_path)
        plt.close()
        images.append(f'images/{col}_distribution.png')

    # Correlation heatmap for numeric columns
    numeric_df = df.select_dtypes(include='number')
    if not numeric_df.empty:
        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
        correlation_image_path = os.path.join('static/images', 'correlation_heatmap.png')
        plt.savefig(correlation_image_path)
        plt.close()
        images.append('images/correlation_heatmap.png')
    
    return images

# Function to identify the target column automatically
def identify_target_column(df):
    candidate_cols = [col for col in df.columns if df[col].nunique() <= 10 and df[col].dtype in ['int64', 'object']]
    if len(candidate_cols) == 1:
        return candidate_cols[0]
    elif candidate_cols:
        return candidate_cols[0]
    else:
        return df.columns[-1]

# Function to perform model training and evaluation
def train_and_evaluate_model(df, target_col, model_choice):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    is_classification = y.nunique() <= 10 and y.dtype == 'int'

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = None
    if is_classification:
        if model_choice == 'Logistic Regression':
            model = LogisticRegression(max_iter=1000)
        elif model_choice == 'Decision Tree':
            model = DecisionTreeClassifier()
        elif model_choice == 'Random Forest':
            model = RandomForestClassifier()
        elif model_choice == 'K-Nearest Neighbors':
            model = KNeighborsClassifier()
    else:
        if model_choice == 'Linear Regression':
            model = LinearRegression()
        elif model_choice == 'Decision Tree':
            model = DecisionTreeRegressor()
        elif model_choice == 'Random Forest':
            model = RandomForestRegressor()
        elif model_choice == 'K-Nearest Neighbors':
            model = KNeighborsRegressor()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if is_classification:
        accuracy = accuracy_score(y_test, y_pred)
        return f'Accuracy: {accuracy * 100:.2f}%'
    else:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return f'Mean Squared Error: {mse:.2f}, R-squared: {r2:.2f}'

# Flask routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        df = pd.read_csv(filepath)
        df = preprocess_data(df)

        target_col = identify_target_column(df)
        images = perform_eda(df)
        return render_template('home.html', images=images, target_col=target_col)

@app.route('/seaborn/<dataset_name>')
def seaborn_dataset(dataset_name):
    df = load_seaborn_dataset(dataset_name)
    if isinstance(df, str):
        return f"Error loading dataset: {df}"
    
    df = preprocess_data(df)
    target_col = identify_target_column(df)
    images = perform_eda(df)
    return render_template('home.html', images=images, target_col=target_col)

if __name__ == '__main__':
    app.run(debug=True)
