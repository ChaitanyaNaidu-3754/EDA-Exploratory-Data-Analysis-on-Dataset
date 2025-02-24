import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2  # OpenCV for image processing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import matplotlib
matplotlib.use('Agg')

# Initialize Flask app
app = Flask(__name__)

# Folder to save uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions for uploaded files
ALLOWED_EXTENSIONS = {'csv'}

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

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
        df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = LabelEncoder().fit_transform(df[col])
    return df


# Ensure pad_token_id is set
tokenizer.pad_token_id = tokenizer.eos_token_id

# Function to generate a description for an image using GPT-2
def generate_description_for_image(image_path,feature_name, max_chars=1000):
    # Open the image using OpenCV
    img = cv2.imread(image_path)

    # Convert the image to grayscale to simplify description
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate mean brightness of the image
    mean_brightness = gray.mean()

    # Generate a description based on the image's characteristics (e.g., mean brightness)
    prompt = f"In a few lines decribe the image and give a prediction. Given the image characteristics, including a mean brightness of {mean_brightness:.2f}, describe the mean, median. Additionally, predict the next value of '{feature_name}' based on the observed trends and any visible patterns in the image."

    # Tokenize the prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)
    attention_mask = inputs.ne(tokenizer.pad_token_id).long()

    # Generate the description from GPT-2
    outputs = model.generate(inputs,
                             attention_mask=attention_mask,
                             max_length=700,
                             num_return_sequences=1,
                             no_repeat_ngram_size=2,
                             num_beams=3,
                             early_stopping=True)

    description = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt by finding its length and slicing the generated text
    prompt_length = len(prompt)  # Get the length of the prompt
    description = description[prompt_length:].strip()  # Strip everything before the description starts

    # Ensure the description does not exceed the character limit
    if len(description) > max_chars:
        description = description[:max_chars].rsplit(' ', 1)[0]  # Cut off at word boundary

    return description.strip()

# Function to apply OpenCV image processing (e.g., resizing, edge detection)
def process_image_with_opencv(image_path):
    # Load the image using OpenCV
    img = cv2.imread(image_path)

    # Resize the image to a standard size (e.g., 800x600)
    img_resized = cv2.resize(img, (800, 600))

    # Convert the image to grayscale (simplifies analysis)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Apply edge detection (Canny)
    edges = cv2.Canny(gray, 100, 200)

    # Create a 3-channel color image with edges in blue (preserving original colors)
    edges_colored = cv2.merge([edges, np.zeros_like(edges), np.zeros_like(edges)])

    # Convert original image to BGR color (ensure it's in the same format)
    img_resized_bgr = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Combine the original image with the edges, adding blue edges while preserving the original image
    img_colored_edges = cv2.addWeighted(img_resized_bgr, 1, edges_colored, 0.5, 0)

    # Save the processed image with the colored edges
    processed_image_path = image_path.replace('.png', '_processed.png')
    cv2.imwrite(processed_image_path, cv2.cvtColor(img_colored_edges, cv2.COLOR_RGB2BGR))  # Save back in BGR format

    return processed_image_path

# Function to perform exploratory data analysis (EDA)
def perform_eda(df):
    # Saving the distribution and correlation plot images
    images = []
    descriptions = []

    # Distribution plots for numerical columns
    for col in df.select_dtypes(include='number').columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f'Distribution of {col}')
        
        # Save the plot
        image_filename = f'{col}_distribution.png'
        image_path = os.path.join('static/images', image_filename)
        plt.savefig(image_path)
        plt.close()

        # Process the image with OpenCV (e.g., edge detection)
        processed_image_path = process_image_with_opencv(image_path)
        
        # Ensure we use the correct URL format
        processed_image_url = processed_image_path.replace(os.path.sep, '/').replace('static/', '')
        images.append(f'images/{processed_image_url.split("/")[-1]}')
        
        # Generate description for the processed image
        description = generate_description_for_image(processed_image_path, col , max_chars=1000)
        descriptions.append(description)

    # Correlation heatmap for numeric columns
    numeric_df = df.select_dtypes(include='number')
    if not numeric_df.empty:
        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
        
        # Save the correlation heatmap
        correlation_image_path = os.path.join('static/images', 'correlation_heatmap.png')
        plt.savefig(correlation_image_path)
        plt.close()

        # Process the heatmap with OpenCV (e.g., edge detection)
        processed_correlation_image_path = process_image_with_opencv(correlation_image_path)
        
        # Use the correct URL format for the processed correlation image
        processed_correlation_image_url = processed_correlation_image_path.replace(os.path.sep, '/').replace('static/', '')
        images.append(f'images/{processed_correlation_image_url.split("/")[-1]}')
        
        # Generate description for the processed heatmap image
        description = generate_description_for_image(processed_correlation_image_path,"correlation matrix", max_chars=700)
        descriptions.append(description)
    
    return images, descriptions

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
        images, descriptions = perform_eda(df)

        # Pair images with descriptions as tuples
        image_desc_pairs = list(zip(images, descriptions))

        # Pass dataset name (filename) to the template
        return render_template('home.html', image_desc_pairs=image_desc_pairs, target_col=target_col, dataset_name=filename)

@app.route('/seaborn/<dataset_name>')
def seaborn_dataset(dataset_name):
    df = load_seaborn_dataset(dataset_name)
    if isinstance(df, str):
        return f"Error loading dataset: {df}"
    
    df = preprocess_data(df)
    target_col = identify_target_column(df)
    images, descriptions = perform_eda(df)

    # Pair images with descriptions as tuples
    image_desc_pairs = list(zip(images, descriptions))

    return render_template('home.html', image_desc_pairs=image_desc_pairs, target_col=target_col, dataset_name=dataset_name)

if __name__ == '__main__':
    app.run(debug=True)