import IPython
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True

df = pd.read_csv('dataset.csv')
df['ST%'] = pd.to_numeric(df['ST%'], errors='coerce')  
df = df[df['ST%'] >= 0.5]
# Preprocess data
label_encoders = {}
for column in ['CATEGORY', 'SEASON', 'FIT', 'PATTERN', 'FABRIC COMPOSITION', 'Denim Wash', 'COLOR']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le

# Splitting the dataset
features = df[['CATEGORY', 'MRP', 'SEASON']]

# Example for 'FIT'
target = df['FIT']

# Splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Training the model
fit_model = RandomForestClassifier(random_state=42)
fit_model.fit(X_train, y_train)

# Predicting and evaluating
predictions = fit_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy for FIT model: {accuracy}')

# Example for 'FIT'
target = df['PATTERN']

# Splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Training the model
pattern_model = RandomForestClassifier(random_state=42)
pattern_model.fit(X_train, y_train)

# Predicting and evaluating
predictions = pattern_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy for FIT model: {accuracy}')

# Target for FABRIC COMPOSITION
target_fabric = df['FABRIC COMPOSITION']

# Splitting data into training and testing for FABRIC COMPOSITION
X_train_fabric, X_test_fabric, y_train_fabric, y_test_fabric = train_test_split(features, target_fabric, test_size=0.3, random_state=42)

# Training the model for FABRIC COMPOSITION
model_fabric = RandomForestClassifier(random_state=42)
model_fabric.fit(X_train_fabric, y_train_fabric)

# Predicting and evaluating for FABRIC COMPOSITION
predictions_fabric = model_fabric.predict(X_test_fabric)
accuracy_fabric = accuracy_score(y_test_fabric, predictions_fabric)
print(f'Accuracy for FABRIC COMPOSITION model: {accuracy_fabric}')

# Target for COLOR
target_color = df['COLOR']

# Splitting data into training and testing for COLOR
X_train_color, X_test_color, y_train_color, y_test_color = train_test_split(features, target_color, test_size=0.3, random_state=42)

# Training the model for COLOR
model_color = RandomForestClassifier(random_state=42)
model_color.fit(X_train_color, y_train_color)

# Predicting and evaluating for COLOR
predictions_color = model_color.predict(X_test_color)
accuracy_color = accuracy_score(y_test_color, predictions_color)
print(f'Accuracy for COLOR model: {accuracy_color}')


# Target for COLOR
target_color = df['Denim Wash']

# Splitting data into training and testing for COLOR
X_train_color, X_test_color, y_train_color, y_test_color = train_test_split(features, target_color, test_size=0.3, random_state=42)

# Training the model for COLOR
model_denim = RandomForestClassifier(random_state=42)
model_denim.fit(X_train_color, y_train_color)

# Predicting and evaluating for COLOR
predictions_color = model_denim.predict(X_test_color)
accuracy_color = accuracy_score(y_test_color, predictions_color)
print(f'Accuracy for COLOR model: {accuracy_color}')


def predict_attributes(category, mrp, season):
    # Encode inputs
    encoded_inputs = []
    for encoder, input_value in zip(['CATEGORY', 'MRP', 'SEASON'], [category, mrp, season]):
        if encoder == 'MRP':  # Assuming MRP does not need encoding
            encoded_inputs.append(input_value)
        else:
            encoded_input = label_encoders[encoder].transform([input_value])[0]
            encoded_inputs.append(encoded_input)

    # Reshape inputs for prediction
    encoded_inputs = [encoded_inputs]  # The model expects a 2D array

    # Predict 'FIT'
    fit_pred = fit_model.predict(encoded_inputs)[0]
    fit_pred = label_encoders['FIT'].inverse_transform([fit_pred])[0]

    # Predict 'PATTERN'
    pattern_pred = pattern_model.predict(encoded_inputs)[0]
    pattern_pred = label_encoders['PATTERN'].inverse_transform([pattern_pred])[0]

    # Predict 'FABRIC COMPOSITION'
    fabric_comp_pred = model_fabric.predict(encoded_inputs)[0]
    fabric_comp_pred = label_encoders['FABRIC COMPOSITION'].inverse_transform([fabric_comp_pred])[0]

    # Predict 'Denim Wash'
    denim_wash_pred = model_denim.predict(encoded_inputs)[0]  # Assuming model_color is for 'Denim Wash'
    denim_wash_pred = label_encoders['Denim Wash'].inverse_transform([denim_wash_pred])[0]

    # Predict 'COLOR'
    color_pred = model_color.predict(encoded_inputs)[0]  # Assuming another model instance for 'COLOR'
    color_pred = label_encoders['COLOR'].inverse_transform([color_pred])[0]

    return {
        'FIT': fit_pred,
        'PATTERN': pattern_pred,
        'FABRIC COMPOSITION': fabric_comp_pred,
        'DENIM WASH': denim_wash_pred,
        'COLOR': color_pred
    }

def predict_top_n_attributes(category, mrp, season, n=2):
    # Encode inputs
    encoded_inputs = []
    for encoder, input_value in zip(['CATEGORY', 'MRP', 'SEASON'], [category, mrp, season]):
        if encoder == 'MRP':  # MRP does not need encoding
            encoded_inputs.append(input_value)
        else:
            encoded_input = label_encoders[encoder].transform([input_value])[0]
            encoded_inputs.append(encoded_input)

    # Reshape inputs for prediction
    encoded_inputs = [encoded_inputs]  # The model expects a 2D array

    predictions = {}

    # Predict 'FIT' with top N predictions
    fit_probs = fit_model.predict_proba(encoded_inputs)[0]
    top_n_fit_indices = fit_probs.argsort()[-n:][::-1]  # Get indices of top N probabilities
    top_n_fit_preds = label_encoders['FIT'].inverse_transform(top_n_fit_indices)
    predictions['FIT'] = top_n_fit_preds.tolist()

    # Predict 'PATTERN' with top N predictions
    pattern_probs = pattern_model.predict_proba(encoded_inputs)[0]
    top_n_pattern_indices = pattern_probs.argsort()[-n:][::-1]
    top_n_pattern_preds = label_encoders['PATTERN'].inverse_transform(top_n_pattern_indices)
    predictions['PATTERN'] = top_n_pattern_preds.tolist()

    # Predict 'FABRIC COMPOSITION' with top N predictions
    fabric_comp_probs = model_fabric.predict_proba(encoded_inputs)[0]
    top_n_fabric_comp_indices = fabric_comp_probs.argsort()[-n:][::-1]
    top_n_fabric_comp_preds = label_encoders['FABRIC COMPOSITION'].inverse_transform(top_n_fabric_comp_indices)
    predictions['FABRIC COMPOSITION'] = top_n_fabric_comp_preds.tolist()

    # Predict 'DENIM WASH' with top N predictions
    denim_wash_probs = model_denim.predict_proba(encoded_inputs)[0]
    top_n_denim_wash_indices = denim_wash_probs.argsort()[-n:][::-1]
    top_n_denim_wash_preds = label_encoders['Denim Wash'].inverse_transform(top_n_denim_wash_indices)
    predictions['DENIM WASH'] = top_n_denim_wash_preds.tolist()

    # Predict 'COLOR' with top N predictions
    color_probs = model_color.predict_proba(encoded_inputs)[0]
    top_n_color_indices = color_probs.argsort()[-n:][::-1]
    top_n_color_preds = label_encoders['COLOR'].inverse_transform(top_n_color_indices)
    predictions['COLOR'] = top_n_color_preds.tolist()

    return predictions


@app.route('/', methods=['GET', 'POST'])
def index():
    import pandas as pd
    df2 = pd.read_csv('dataset.csv')
    # Load unique values for dropdowns
    unique_categories = sorted(df2['CATEGORY'].unique().tolist())
    unique_mrps = sorted(df2['MRP'].unique().astype(str).tolist())  # Convert MRP to string for consistency in dropdown
    unique_seasons = sorted(df2['SEASON'].unique().tolist())

    print(unique_categories)

    if request.method == 'POST':
        category = request.form['category']
        mrp = request.form['mrp']
        season = request.form['season']
        predictions = predict_top_n_attributes(category, mrp, season, n=2)
        print(predictions)
        return render_template('index.html', predictions=predictions, unique_categories=unique_categories, unique_mrps=unique_mrps, unique_seasons=unique_seasons)
    return render_template('index.html', unique_categories=unique_categories, unique_mrps=unique_mrps, unique_seasons=unique_seasons)

