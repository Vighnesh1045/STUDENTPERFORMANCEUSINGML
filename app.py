from flask import Flask, request, render_template, redirect, session, url_for
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'

# Define User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False, default='default_username')
    email = db.Column(db.String(100), nullable=True)
    password = db.Column(db.String(100))

    def __init__(self, email, password, name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

with app.app_context():
    db.create_all()

# Define routes
@app.route('/')
def index():
    return 'hi'

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        
        new_user = User(name=name, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')
        
    return render_template("register.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password=password):  
            session['email'] = user.email
            return redirect('/dashboard')
        else:
            return render_template('login.html', error='Invalid email or password')
        
    return render_template("login.html")

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'email' in session:
        email = session['email']
        user = User.query.filter_by(email=session['email']).first()
        return render_template('dashboard.html', user=user)
    else:
        return redirect(url_for('login'))

@app.route('/prediction1', methods=['GET', 'POST'])
def prediction1():
    if request.method == 'POST':
        return redirect(url_for('prediction2'))
    return render_template('prediction1.html')

colleges = {
    "Mumbai": ["St. Xavier's College", "Jai Hind College", "Mithibai College", "HR College of Commerce and Economics", "R.A. Podar College of Commerce and Economics", "Sophia College for Women", "Ruia College"],
    "Navi Mumbai": ["Karmaveer Bhaurao Patil College", "F.G. Naik College Of Arts Science(IT) & Commerce", "Sanpada College Of Commerce And Technology", "SIES College of Arts, Science and Commerce Nerul", "Rajiv Gandhi College Of Arts Commerce & Science"],
    "Thane": ["Joshi Bedekar College", "Hill Spring International Junior College Of Science & Commerce.", "Suraju Singh Memorial College Of Education & Research (D Ed & B Ed)", "Reena Mehta College.", "Seth Hirachand Mutha College of Arts, Commerce and Science, Thane"],
    "Pune": ["Fergusson College", "Abasaheb Garware College", "SP College", "St. Mira's College for Girls", "Brihan Maharashtra College of Commerce (BMCC)"]
}

@app.route('/view_clg')
def home():
    return render_template('view_clg.html', cities=colleges.keys())

@app.route('/city/<city>')
def city_collages(city):
    city_colleges = {
        'Mumbai': {
            'St. Xavier\'s College': 'https://xaviers.ac/',
            'Jai Hind College': 'https://www.jaihindcollege.com/',
            'Mithibai College': 'https://mithibai.ac.in/',
            'HR College of Commerce and Economics': 'https://www.hrcollege.edu/',
            'R.A. Podar College of Commerce and Economics': 'https://www.rapodar.ac.in/',
            'Sophia College for Women': 'https://sophiacollegemumbai.com/',
            'Ruia College': 'https://www.ruiacollege.edu/'
        },
        'Navi Mumbai': {
            'Karmaveer Bhaurao Patil College': 'https://www.kbpcollegevashi.edu.in/',
            'F.G. Naik College Of Arts Science(IT) & Commerce': 'https://www.fgnaikcollege.com/',
            'Sanpada College Of Commerce And Technology': 'https://scct.edu.in/',
            'SIES College of Arts, Science and Commerce Nerul': 'https://www.siesascn.edu.in/',
            'Rajiv Gandhi College Of Arts Commerce & Science': 'https://setrgc.edu.in/'
        },
        'Thane': {
            'Joshi Bedekar College': 'https://www.joshibedekar.org/',
            'Hill Spring International Junior College Of Science & Commerce.': 'https://www.hsieducation.org/',
            'Suraju Singh Memorial College Of Education & Research (D Ed & B Ed)': 'https://www.ssmbedcollege.com/',
            'Reena Mehta College.': 'https://rmc.edu.in/',
            'Seth Hirachand Mutha College of Arts, Commerce and Science, Thane': 'https://shmutha.org/'
        },
        'Pune': {
            'Fergusson College': 'https://www.fergusson.edu/',
            'Abasaheb Garware College': 'https://garwarecollege.mespune.in/',
            'SP College': 'https://www.spcollegepune.ac.in/',
            'St. Mira\'s College for Girls': 'https://www.stmirascollegepune.edu.in/',
            'Brihan Maharashtra College of Commerce (BMCC)': 'https://www.bmcc.ac.in/'
        }
    }
    return render_template('city_collages.html', colleges=city_colleges.get(city, {}))

# Load and prepare the model
with open('stream_predict_final.pkl', 'rb') as file:
    best_model = pickle.load(file)

df = pd.read_excel("studentmarksheetupdated.xlsx")
df['Branch'] = df['Branch'].replace({'Commerce': 'Commerce/Arts', 'Arts': 'Commerce/Arts'})
X = df.drop(['Branch', 'Names'], axis=1)
y = df['Branch']

encoder = OneHotEncoder(sparse_output=False, drop='first')
X_encoded = pd.concat([X, pd.DataFrame(encoder.fit_transform(X[['Gender']]), columns=['Gender_N'])], axis=1)
X_encoded = X_encoded.drop('Gender', axis=1)

scaler = MinMaxScaler()
subject_columns = ['Maths', 'Physics', 'Chemistry', 'English', 'Biology', 'Economics', 'History', 'Civics']
X_encoded[subject_columns] = scaler.fit_transform(X[subject_columns])

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f'Best Hyperparameters: {best_params}')

best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy with best model: {accuracy}')

conf_matrix = confusion_matrix(y_test, predictions)
print(f'Confusion Matrix:\n{conf_matrix}')

class_report = classification_report(y_test, predictions)
print(f'Classification Report:\n{class_report}')

feature_importances = pd.DataFrame({'Feature': X_test.columns, 'Importance': best_model.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
print('Feature Importances:')
print(feature_importances)

@app.route('/prediction2', methods=['GET', 'POST'])
def prediction2():
    if request.method == 'POST':
        gender = request.form['gender']
        maths = int(request.form['maths'])
        physics = int(request.form['physics'])
        chemistry = int(request.form['chemistry'])
        english = int(request.form['english'])
        biology = int(request.form['biology'])
       
