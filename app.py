from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split



app = Flask(__name__)

# Load the saved model
with open('C:\\Users\\Vighnesh\\Desktop\\ML Project\\stream_predict_final.pkl', 'rb') as file:
    best_model = pickle.load(file)

# Define the scaler
scaler = MinMaxScaler()

# Define the subject columns
subject_columns = ['Maths', 'Physics', 'Chemistry', 'English', 'Biology', 'Economics', 'History', 'Civics']

# Load the data
df = pd.read_excel("studentmarksheetupdated.xlsx")

# Combine "Commerce" and "Arts" into a single category "Commerce/Arts"
df['Branch'] = df['Branch'].replace({'Commerce': 'Commerce/Arts', 'Arts': 'Commerce/Arts'})

# Extract features and target
X = df.drop(['Branch', 'Names'], axis=1)  # Drop 'Name' column
y = df['Branch']

# One-hot encode 'Gender' column
encoder = OneHotEncoder(sparse=False, drop='first')  # Drop first to avoid multicollinearity
X_encoded = pd.concat([X, pd.DataFrame(encoder.fit_transform(X[['Gender']]), columns=['Gender_N'])], axis=1)

# Drop the original 'Gender' column
X_encoded = X_encoded.drop('Gender', axis=1)

# Min-Max scale subject columns
X_encoded[subject_columns] = scaler.fit_transform(X[subject_columns])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    gender = request.form['gender']
    maths = int(request.form['maths'])
    physics = int(request.form['physics'])
    chemistry = int(request.form['chemistry'])
    english = int(request.form['english'])
    biology = int(request.form['biology'])
    economics = int(request.form['economics'])
    history = int(request.form['history'])
    civics = int(request.form['civics'])

    # Create a DataFrame from the input
    input_data = {
        'Gender_N': int(gender),  # Update according to your data
        'Maths': maths,
        'Physics': physics,
        'Chemistry': chemistry,
        'English': english,
        'Biology': biology,
        'Economics': economics,
        'History': history,
        'Civics': civics
    }
    input_df = pd.DataFrame([input_data])

    # Reorder the columns to match the order during training
    input_df = input_df[X_train.columns]

    # Preprocess the input data
    input_df[subject_columns] = scaler.transform(input_df[subject_columns])

    # Return the predicted branch
    prediction = best_model.predict(input_df)
    # return render_template('index.html', branch=prediction[0])

     # Return the predicted branch in a new HTML template
    return render_template('prediction_result.html', branch=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
