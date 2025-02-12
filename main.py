from flask import Flask, request, render_template, jsonify, send_from_directory  # Import jsonify and send_from_directory
import numpy as np
import pandas as pd
import pickle
from flask_cors import CORS  # Add CORS support
from models import db, Diagnosis
import os
import google.generativeai as genai
from dotenv import load_dotenv
import asyncio
import threading
from sqlalchemy import desc
from difflib import SequenceMatcher
try:
    import nltk
    from nltk.corpus import wordnet  # Change this line
except ImportError:
    print("NLTK not installed. Basic similarity matching will be used.")
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import joblib
from sklearn.svm import SVC
from sqlalchemy import inspect

# Global variables
sym_des = None
precautions = None
workout = None
description = None
medications = None
diets = None
diseases_list = {
    'Fungal infection': 'Fungal infection',
    'Allergy': 'Allergy',
    'GERD': 'GERD',
    'Chronic cholestasis': 'Chronic cholestasis',
    'Drug Reaction': 'Drug Reaction',
    'Peptic ulcer diseae': 'Peptic ulcer diseae',
    'AIDS': 'AIDS',
    'Diabetes': 'Diabetes',
    'Gastroenteritis': 'Gastroenteritis',
    'Bronchial Asthma': 'Bronchial Asthma',
    'Hypertension': 'Hypertension',
    'Migraine': 'Migraine',
    'Cervical spondylosis': 'Cervical spondylosis',
    'Paralysis (brain hemorrhage)': 'Paralysis (brain hemorrhage)',
    'Jaundice': 'Jaundice',
    'Malaria': 'Malaria',
    'Chicken pox': 'Chicken pox',
    'Dengue': 'Dengue',
    'Typhoid': 'Typhoid',
    'hepatitis A': 'hepatitis A',
    'Hepatitis B': 'Hepatitis B',
    'Hepatitis C': 'Hepatitis C',
    'Hepatitis D': 'Hepatitis D',
    'Hepatitis E': 'Hepatitis E',
    'Alcoholic hepatitis': 'Alcoholic hepatitis',
    'Tuberculosis': 'Tuberculosis',
    'Common Cold': 'Common Cold',
    'Pneumonia': 'Pneumonia',
    'Dimorphic hemmorhoids(piles)': 'Dimorphic hemmorhoids(piles)',
    'Heart attack': 'Heart attack',
    'Varicose veins': 'Varicose veins',
    'Hypothyroidism': 'Hypothyroidism',
    'Hyperthyroidism': 'Hyperthyroidism',
    'Hypoglycemia': 'Hypoglycemia',
    'Osteoarthristis': 'Osteoarthristis',
    'Arthritis': 'Arthritis',
    '(vertigo) Paroymsal  Positional Vertigo': '(vertigo) Paroymsal  Positional Vertigo',
    'Acne': 'Acne',
    'Urinary tract infection': 'Urinary tract infection',
    'Psoriasis': 'Psoriasis',
    'Impetigo': 'Impetigo'
}

print("Starting MedMentors application...")

# First, initialize Flask app
app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medmentors.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key'
db.init_app(app)

# Initialize NewsAPI
newsapi = NewsApiClient(api_key='07f0e4fe26d74db5b6f8a881a7434e55')

# Function definitions
def load_datasets():
    """Load all required datasets"""
    global sym_des, precautions, workout, description, medications, diets
    try:
        print("Loading datasets...")
        sym_des = pd.read_csv("training_data/symtoms_df.csv")
        precautions = pd.read_csv("training_data/precautions_df.csv")
        workout = pd.read_csv("training_data/workout_df.csv")
        description = pd.read_csv("training_data/description.csv")
        medications = pd.read_csv("training_data/medications.csv")
        diets = pd.read_csv("training_data/diets.csv")
        print("All datasets loaded successfully!")
    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
        raise

# Add this after your imports and before other function definitions
def verify_static_files():
    required_files = [
        'images/medmentors-logo.webp',
        'images/ai-healthcare.jpg'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join('static', file)):
            missing_files.append(file)
    
    if missing_files:
        print("Warning: Missing static files:", missing_files)
        print("Please run setup_static.py to copy required files")

# Update the NLTK setup function
def setup_nltk():
    """Setup NLTK and download required data"""
    try:
        # Try to use wordnet to check if it's downloaded
        wordnet.synsets('test')
    except LookupError:
        # If not downloaded, download required NLTK data
        print("Downloading required NLTK data...")
        try:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            print("NLTK setup complete!")
        except Exception as e:
            print(f"Error downloading NLTK data: {str(e)}")
            print("Continuing without WordNet similarity...")
    except Exception as e:
        print(f"NLTK setup error: {str(e)}")
        print("Continuing without WordNet similarity...")

# Move these function definitions to the top of your file, after imports
def init_db():
    try:
        with app.app_context():
            db.drop_all()
            db.create_all()
            print("Database initialized successfully!")
    except Exception as e:
        print(f"Database initialization error: {str(e)}")
        pass

def load_model():
    try:
        model_path = 'models/svc.pkl'
        print(f"Attempting to load model from {model_path}")  # Debug print
        
        if not os.path.exists(model_path):
            print("Model file not found. Training new model...")
            # Train a new model
            X, y = prepare_training_data()
            model = SVC(probability=True)  # Set probability=True
            model.fit(X, y)
            
            # Save the model
            os.makedirs('models', exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print("New model trained and saved successfully")
            return model
            
        # Load existing model
        model = pickle.load(open(model_path, 'rb'))
        if not hasattr(model, 'predict_proba'):
            print("Existing model doesn't support probabilities. Training new model...")
            # Train a new model with probability support
            X, y = prepare_training_data()
            model = SVC(probability=True)
            model.fit(X, y)
            
            # Save the new model
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print("New model trained and saved successfully")
        
        print("Model loaded successfully")
        return model
        
    except Exception as e:
        print(f"Error loading/training model: {str(e)}")
        raise

# Add this function to prepare training data
def prepare_training_data():
    """Prepare training data for the model"""
    try:
        # Load your training data
        training_data = pd.read_csv("training_data/Training.csv")
        
        if 'prognosis' not in training_data.columns:
            raise ValueError("Training data does not contain 'prognosis' column")
        
        # Prepare features (X) and target (y)
        X = training_data.drop('prognosis', axis=1)
        y = training_data['prognosis']
        
        print(f"Training data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    except Exception as e:
        print(f"Error preparing training data: {str(e)}")
        raise

# Initialize everything
try:
    # Load all datasets
    load_datasets()
    
    # Call initialization functions
    init_db()
    setup_nltk()

    # Load/train model
    print("Loading/training model...")
    svc = load_model()
    print("Model loaded successfully!")

except Exception as e:
    print(f"Initialization error: {str(e)}")
    raise

#============================================================
# custome and helping functions
#==========================helper funtions================
def helper(dis):
    try:
        # Get description
        desc_row = description[description['Disease'] == dis]
        desc = desc_row['Description'].iloc[0] if not desc_row.empty else ""

        # Get precautions
        pre_row = precautions[precautions['Disease'] == dis]
        pre = pre_row[['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values[0].tolist() if not pre_row.empty else []

        # Get medications
        med_row = medications[medications['Disease'] == dis]
        med = med_row['Medication'].tolist() if not med_row.empty else []

        # Get diet recommendations
        diet_row = diets[diets['Disease'] == dis]
        die = diet_row['Diet'].tolist() if not diet_row.empty else []

        # Get workout recommendations
        workout_row = workout[workout['disease'] == dis]
        wrkout = workout_row['workout'].iloc[0] if not workout_row.empty else ""

        return desc, pre, med, die, wrkout

    except Exception as e:
        print(f"Error in helper function for disease '{dis}': {str(e)}")
        import traceback
        print(traceback.format_exc())
        return "", [], [], [], ""

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

# Add this after your imports
DISEASE_MAPPINGS = {
    # Common variations and synonyms
    'Gastroenteritis': 'Gastritis',
    'Stomach Flu': 'Gastritis',
    'Flu': 'Common Cold',
    'Influenza': 'Common Cold',
    'Upper Respiratory Infection': 'Common Cold',
    'UTI': 'Urinary tract infection',
    # Add more mappings as needed
}

# Add these disease patterns
DISEASE_PATTERNS = {
    "Dengue": {
        "required": ["high_fever"],
        "supporting": ["joint_pain", "pain_behind_the_eyes", "nausea", "headache", "vomiting", "skin_rash"],
        "min_supporting": 2
    },
    "Fungal infection": {
        "required": ["itching"],
        "supporting": ["skin_rash", "nodal_skin_eruptions", "dischromic_patches"],
        "min_supporting": 1
    },
    "Impetigo": {
        "required": ["skin_rash"],
        "supporting": ["high_fever", "muscle_weakness", "joint_pain"],
        "min_supporting": 2,
        "exclude": ["itching"]  # If itching is present, likely not impetigo
    }
    # Add more patterns as needed
}

def check_disease_pattern(symptoms, disease):
    """Check if symptoms match a disease pattern"""
    if disease not in DISEASE_PATTERNS:
        return False
    
    pattern = DISEASE_PATTERNS[disease]
    
    # Check required symptoms
    if not all(sym in symptoms for sym in pattern["required"]):
        return False
        
    # Check for excluded symptoms if any
    if "exclude" in pattern and any(sym in symptoms for sym in pattern["exclude"]):
        return False
    
    # Count supporting symptoms
    supporting_count = sum(1 for sym in pattern["supporting"] if sym in symptoms)
    return supporting_count >= pattern["min_supporting"]

def get_quick_prediction(symptoms):
    """Get immediate prediction from SVC model with pattern validation"""
    try:
        # If single symptom is passed as string, convert to list
        if isinstance(symptoms, str):
            symptoms = [symptoms]
            
        # First check against known patterns
        for disease, pattern in DISEASE_PATTERNS.items():
            if check_disease_pattern(symptoms, disease):
                initial_description, precautions, medications, rec_diet, workout = helper(disease)
                return {
                    "predicted_disease": disease,
                    "description": initial_description,
                    "precautions": precautions[0],
                    "medications": medications,
                    "diet": rec_diet,
                    "workout": workout,
                    "confidence": "High (Pattern Match)"
                }
        
        # If no pattern match, use SVC model
        initial_disease = get_predicted_value(symptoms)
        
        # Validate the SVC prediction against patterns
        if initial_disease in DISEASE_PATTERNS and not check_disease_pattern(symptoms, initial_disease):
            # Look for alternative diagnoses
            for disease, pattern in DISEASE_PATTERNS.items():
                if check_disease_pattern(symptoms, disease):
                    initial_disease = disease
                    break
        
        initial_description, precautions, medications, rec_diet, workout = helper(initial_disease)
        
        return {
            "predicted_disease": initial_disease,
            "description": initial_description,
            "precautions": precautions[0],
            "medications": medications,
            "diet": rec_diet,
            "workout": workout
        }
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return {
            "predicted_disease": "Unrecognized Condition",
            "description": "The condition could not be precisely matched with our database.",
            "precautions": ["Please consult a healthcare professional"],
            "medications": ["Consultation required"],
            "diet": ["Follow a balanced diet"],
            "workout": ["As advised by your doctor"]
        }

def get_similarity_score(str1, str2):
    """Calculate similarity between two disease names"""
    try:
        # Direct string similarity
        sequence_similarity = SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
        
        # Semantic similarity using WordNet if available
        try:
            syns1 = wordnet.synsets(str1)
            syns2 = wordnet.synsets(str2)
            if syns1 and syns2:
                semantic_similarity = syns1[0].wup_similarity(syns2[0]) or 0
                # Combine both similarities (weighted average)
                final_similarity = (sequence_similarity * 0.6) + (semantic_similarity * 0.4)
            else:
                # Fallback to just sequence similarity
                final_similarity = sequence_similarity
        except Exception as e:
            print(f"WordNet similarity error: {str(e)}")
            # Fallback to just sequence similarity
            final_similarity = sequence_similarity
        
        return final_similarity * 100  # Convert to percentage
    except Exception as e:
        print(f"Similarity calculation error: {str(e)}")
        # Fallback to basic string matching
        return 100 if str1.lower() == str2.lower() else 0

# Add this helper function to get disease description
def get_disease_description(disease_name):
    """Get description for a disease from the dataset"""
    try:
        # Look up in description DataFrame
        disease_desc = description[description['Disease'] == disease_name]['Description'].values
        if len(disease_desc) > 0:
            return disease_desc[0]
        return "Description not available"
    except Exception as e:
        print(f"Error getting description for {disease_name}: {str(e)}")
        return "Description not available"

def validate_with_llm(symptoms, ml_prediction, ml_confidence, description_data):
    try:
        symptoms_text = ", ".join(symptoms)
        
        # Create a more comprehensive prompt for disease prediction
        prompt = f"""As a medical diagnostic AI, analyze these symptoms carefully: {symptoms_text}

Current ML Model Prediction: {ml_prediction} (Confidence: {ml_confidence:.2f}%)

Task:
1. Analyze the symptoms independently first
2. Consider these factors:
   - Symptom combinations and their typical patterns
   - Primary vs secondary symptoms
   - Common medical conditions matching these symptoms
3. Compare your analysis with the ML prediction
4. If your prediction differs from ML and you have high confidence, explain why

Dataset Disease Patterns to consider:
- Joint pain + muscle pain often indicates: Arthritis, Fibromyalgia
- Weight gain with joint pain may indicate: Hypothyroidism, Osteoarthritis
- UTI typically presents with: urinary symptoms, burning, frequent urination
- Dengue presents with: high fever, joint pain, eye pain, nausea
- Thyroid issues show: weight changes, fatigue, muscle weakness

Please provide your analysis in this format:
1. Independent Diagnosis: [your predicted disease]
2. Confidence Level: [percentage]
3. Reasoning: [detailed explanation of symptom analysis]
4. Agreement with ML: [yes/no with explanation]
5. Final Recommendation: [which prediction to use and why]

Base your analysis on medical knowledge and symptom patterns."""

        try:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            
            if response and response.text:
                sections = response.text.split('\n')
                llm_result = {
                    "validated_diagnosis": ml_prediction,
                    "confidence_level": f"{ml_confidence:.2f}%",
                    "reasoning": "",
                    "additional_considerations": ""
                }

                llm_diagnosis = ""
                llm_confidence = 0
                agreement_with_ml = False

                for section in sections:
                    if "Independent Diagnosis:" in section:
                        llm_diagnosis = section.split(":")[1].strip()
                    elif "Confidence Level:" in section:
                        conf = section.split(":")[1].strip()
                        llm_confidence = float(''.join(filter(str.isdigit, conf)))
                    elif "Reasoning:" in section:
                        llm_result["reasoning"] = section.split(":")[1].strip()
                    elif "Agreement with ML:" in section:
                        agreement_text = section.split(":")[1].strip().lower()
                        agreement_with_ml = "yes" in agreement_text
                    elif "Final Recommendation:" in section:
                        llm_result["additional_considerations"] = section.split(":")[1].strip()

                # Decision logic
                if not agreement_with_ml and llm_confidence > 70:
                    # LLM strongly disagrees with ML and has high confidence
                    return {
                        "validated_diagnosis": llm_diagnosis,
                        "confidence_level": f"{llm_confidence:.2f}%",
                        "reasoning": llm_result["reasoning"],
                        "additional_considerations": "LLM override due to strong symptom pattern match"
                    }
                elif ml_confidence < 30 and llm_confidence > 60:
                    # ML has very low confidence, but LLM is confident
                    return {
                        "validated_diagnosis": llm_diagnosis,
                        "confidence_level": f"{llm_confidence:.2f}%",
                        "reasoning": llm_result["reasoning"],
                        "additional_considerations": "Using LLM prediction due to low ML confidence"
                    }
                else:
                    # Use ML prediction when confidence is higher or there's agreement
                    return {
                        "validated_diagnosis": ml_prediction,
                        "confidence_level": f"{ml_confidence:.2f}%",
                        "reasoning": "ML prediction confirmed by symptom analysis",
                        "additional_considerations": llm_result["additional_considerations"]
                    }

        except Exception as e:
            print(f"LLM API error: {str(e)}")
            return {
                "validated_diagnosis": ml_prediction,
                "confidence_level": f"{ml_confidence:.2f}%",
                "reasoning": "Using ML prediction due to LLM error",
                "additional_considerations": f"Error: {str(e)}"
            }

    except Exception as e:
        print(f"Validation error: {str(e)}")
        return {
            "validated_diagnosis": ml_prediction,
            "confidence_level": f"{ml_confidence:.2f}%",
            "reasoning": "Using ML prediction due to validation error",
            "additional_considerations": f"Error: {str(e)}"
        }

# Temporary storage for validation results
validation_results = {}

# Then define your routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            symptoms = request.form.get('symptoms', '').strip()
            if not symptoms:
                return render_template('index.html', 
                                    error="Please enter at least one symptom",
                                    symptoms_dict=symptoms_dict)

            user_symptoms = [s.strip() for s in symptoms.split(',')]
            user_symptoms = [s for s in user_symptoms if s in symptoms_dict]

            if not user_symptoms:
                return render_template('index.html', 
                                    error="Invalid symptoms. Please enter valid symptoms.",
                                    symptoms_dict=symptoms_dict)

            # Get ML prediction
            input_vector = np.zeros(len(symptoms_dict))
            for symptom in user_symptoms:
                if symptom in symptoms_dict:
                    input_vector[symptoms_dict[symptom]] = 1

            ml_prediction = svc.predict([input_vector])[0]
            try:
                ml_confidence = np.max(svc.predict_proba([input_vector])) * 100
            except:
                ml_confidence = 80

            # Get LLM validation
            validation_result = validate_with_llm(user_symptoms, ml_prediction, ml_confidence, description)
            
            final_prediction = validation_result['validated_diagnosis']
            confidence = float(validation_result['confidence_level'].rstrip('%'))

            # Get additional information
            description_text, precautions, medications, diet, workout = helper(final_prediction)

            # Save to database
            try:
                print("Attempting to save diagnosis to database...")
                print(f"Symptoms: {symptoms}")
                print(f"Final Prediction: {final_prediction}")
                print(f"Confidence: {confidence:.2f}%")
                
                diagnosis = Diagnosis(
                    symptoms=symptoms,
                    predicted_disease=final_prediction,
                    confidence_level=f"{confidence:.2f}%",
                    description=description_text,
                    ml_prediction=ml_prediction,
                    llm_prediction=final_prediction
                )
                db.session.add(diagnosis)
                db.session.commit()
                print(f"Successfully saved diagnosis to database with ID: {diagnosis.id}")
            except Exception as e:
                print(f"Database error: {str(e)}")
                print(f"Full error: {traceback.format_exc()}")
                db.session.rollback()

            return render_template('result.html',
                                symptoms=user_symptoms,
                                disease=final_prediction,
                                confidence=f"{confidence:.2f}%",
                                description=description_text,
                                precautions=precautions,
                                medications=medications,
                                diet=diet,
                                workout=workout,
                                ml_prediction=ml_prediction,
                                ml_confidence=f"{ml_confidence:.2f}%",
                                reasoning=validation_result.get('reasoning', ''))

        except Exception as e:
            print(f"Error processing symptoms: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return render_template('index.html',
                                error=f"An error occurred: {str(e)}",
                                symptoms_dict=symptoms_dict)

    return render_template('index.html', symptoms_dict=symptoms_dict)

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/developer')
def developer():
    return render_template("developer.html")

@app.route('/blog')
def blog():
    return render_template("blog.html")

@app.route('/news')
def news():
    try:
        # Get date range (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Format dates for the API
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')
        
        # Fetch medical news with specific keywords and sorting
        news_data = newsapi.get_everything(
            q='medical OR healthcare OR medicine OR disease OR treatment',
            language='en',
            from_param=from_date,
            to=to_date,
            sort_by='publishedAt',
            domains='medicalnewstoday.com,health.harvard.edu,mayoclinic.org,webmd.com,medscape.com',
            page_size=30
        )
        
        # Process and sort articles
        articles = news_data.get('articles', [])
        
        # Convert string dates to datetime objects for proper sorting
        for article in articles:
            article['publishedAt'] = datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
        
        # Sort articles by date (newest first)
        articles.sort(key=lambda x: x['publishedAt'], reverse=True)
        
        # Convert dates back to readable format
        for article in articles:
            article['publishedAt'] = article['publishedAt'].strftime('%B %d, %Y')
        
        return render_template('news.html', 
                             news_articles=articles,
                             current_time=datetime.now().strftime('%B %d, %Y %H:%M:%S'))
                             
    except Exception as e:
        print(f"News API Error: {str(e)}")  # Log the error
        return render_template('news.html', 
                             error="Unable to fetch news at this time. Please try again later.",
                             news_articles=[],
                             current_time=datetime.now().strftime('%B %d, %Y %H:%M:%S'))

@app.route('/history')
def history():
    try:
        print("Fetching diagnosis history...")
        diagnoses = Diagnosis.query.order_by(Diagnosis.date_created.desc()).all()
        print(f"Found {len(diagnoses)} diagnoses")
        for d in diagnoses:
            print(f"Diagnosis: {d.predicted_disease}, Date: {d.date_created}, Symptoms: {d.symptoms}")
        return render_template('history.html', diagnoses=diagnoses)
    except Exception as e:
        print(f"Error fetching history: {str(e)}")
        print(f"Full error: {traceback.format_exc()}")
        return render_template('history.html', error="Error fetching history", diagnoses=[])

# Update the fetch_news_articles function
def fetch_news_articles():
    try:
        print("Starting news fetch...")  # Debug print
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        print("Calling NewsAPI...")  # Debug print
        news_response = newsapi.get_everything(
            q='healthcare OR medicine OR medical research',
            language='en',
            from_param=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d'),
            sort_by='publishedAt',
            page_size=12
        )
        
        print(f"NewsAPI response status: {news_response['status']}")  # Debug print
        
        if news_response['status'] == 'ok':
            articles = news_response['articles']
            print(f"Retrieved {len(articles)} articles")  # Debug print
            
            # Process and format the articles
            formatted_articles = []
            for article in articles:
                formatted_article = {
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'image': article.get('urlToImage', ''),
                    'publishedAt': datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ').strftime('%B %d, %Y'),
                    'source': {'name': article.get('source', {}).get('name', 'Unknown Source')},
                    'url': article.get('url', '#')
                }
                formatted_articles.append(formatted_article)
            
            return formatted_articles
            
    except Exception as e:
        print(f"Error fetching news: {str(e)}")
        # Return fallback news if API fails
        return [{
            'title': 'Unable to fetch latest news',
            'description': 'Please try again later.',
            'image': None,
            'publishedAt': datetime.now().strftime('%B %d, %Y'),
            'source': {'name': 'System Message'},
            'url': '#'
        }]

if __name__ == '__main__':
    try:
        print("Initializing application components...")
        verify_static_files()
        load_datasets()
        setup_nltk()
        
        print("Initializing database...")
        with app.app_context():
            # Drop all tables and recreate them
            db.drop_all()
            db.create_all()
            print("Database tables created successfully!")
            
            # Verify tables exist
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            print(f"Database tables: {tables}")
            
        print("Starting server...")
        app.run(host='127.0.0.1', port=5000, debug=True)
    except Exception as e:
        print(f"Server startup error: {str(e)}")
        import traceback
        print(traceback.format_exc())

@app.context_processor
def inject_symptoms():
    return dict(symptoms_dict=symptoms_dict)

# Add error handling
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

# Add API versioning
@app.route("/api/v1/predict", methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', [])
        
        if not symptoms:
            return jsonify({"error": "No symptoms provided"}), 400

        # Get enhanced prediction with LLM validation
        prediction = get_enhanced_prediction(symptoms)
        
        return jsonify(prediction)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def process_symptoms(symptoms_input):
    """Standardize symptom input"""
    # Mapping of common variations to standard symptom names
    symptom_mapping = {
        "fever": "high_fever",
        "high fever": "high_fever",
        "eye pain": "pain_behind_the_eyes",
        "pain behind eyes": "pain_behind_the_eyes",
        "joint pain": "joint_pain",
        "muscle pain": "muscle_pain",
        "head ache": "headache",
        "head pain": "headache",
        # Add more mappings as needed
    }
    
    processed_symptoms = []
    for symptom in symptoms_input:
        # Clean the symptom text
        cleaned = symptom.lower().strip()
        # Map to standard name if exists
        standard_name = symptom_mapping.get(cleaned, cleaned)
        # Replace spaces with underscores
        standard_name = standard_name.replace(" ", "_")
        processed_symptoms.append(standard_name)
    
    return processed_symptoms

# Add these new routes
@app.route('/reset', methods=['POST'])
def reset():
    try:
        Diagnosis.query.delete()
        db.session.commit()
        return jsonify({'message': 'History cleared successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

def migrate_db():
    """
    Migrate the database when schema changes
    """
    try:
        with app.app_context():
            # Add any data migration logic here if needed
            db.create_all()
            print("Database migration completed successfully!")
    except Exception as e:
        print(f"Error during migration: {str(e)}")

# Add this route for admin use
@app.route('/admin/migrate')
def admin_migrate():
    try:
        migrate_db()
        return jsonify({"message": "Database migration completed successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Update the static file handling
@app.route('/static/<path:filename>')
def serve_static(filename):
    try:
        return send_from_directory('static', filename)
    except Exception as e:
        print(f"Error serving static file {filename}: {str(e)}")
        return '', 404  # Return 404 if file not found

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"}), 200