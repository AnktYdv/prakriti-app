import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Load datasets
try:
    data_features = pd.read_csv('fold_all_c.csv')
    data_folds = pd.read_csv('fold_all_class.csv')
except FileNotFoundError:
    st.error("Error: Please ensure 'fold_all_c.csv' and 'fold_all_class.csv' are uploaded to the app directory.")
    st.stop()

# Preprocess data
features = data_features.drop(columns=['SampleID', 'C'])
labels = data_features['C']
folds = data_folds[['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5', 
                    'Fold6', 'Fold7', 'Fold8', 'Fold9', 'Fold10']]

# Encode categorical features
label_encoders = {}
for column in features.columns:
    if features[column].dtype == 'object':
        le = LabelEncoder()
        features[column] = le.fit_transform(features[column].astype(str))
        label_encoders[column] = le

# Encode labels
le_label = LabelEncoder()
y_encoded = le_label.fit_transform(labels)

# Train final model
model = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', random_state=42)
model.fit(features, y_encoded)

# Define questionnaire
questionnaire = [
    {'question': 'What is your gender?', 'feature': 'GENDER', 'options': ['Male', 'Female']},
    {'question': 'What is your body frame?', 'feature': 'F1', 'options': ['Narrow', 'Medium', 'Wide']},
    {'question': 'What is your body build (bulk)?', 'feature': 'F2', 'options': ['Weakly developed', 'Moderately developed', 'Well developed']},
    {'question': 'What is your body build (musculature)?', 'feature': 'F3', 'options': ['Thin Musculature', 'Soft and Loosely knitted Musculature', 'Smooth and Firmly knitted Musculature']},
    {'question': 'What is your forehead length?', 'feature': 'F4', 'options': ['Small', 'Medium', 'Large']},
    {'question': 'What is your nails texture?', 'feature': 'F5', 'options': ['Smooth', 'Soft', 'Rough']},
    {'question': 'What is your nails colour?', 'feature': 'F6', 'options': ['Reddish', 'Pale', 'Pink']},
    {'question': 'What is your finger nail size?', 'feature': 'F7', 'options': ['Small', 'Medium', 'Large']},
    {'question': 'What is your skin appearance?', 'feature': 'F8', 'options': ['Cracked', 'Lustrous', 'Rough', 'Moles', 'Marks', 'Pimples', 'Freckles', 'Wrinkles', 'None']},
    {'question': 'What is your skin colour/complexion?', 'feature': 'F9', 'options': ['Fair with reddish tinge', 'Fair with yellowish tinge', 'Dark', 'Dusky', 'Wheatish', 'Fair with pale tinge', 'Fair with pink tinge']},
    {'question': 'What is your skin nature?', 'feature': 'F10', 'options': ['Dry', 'Oily', 'Normal', 'Seasonal']},
    {'question': 'What is your skin texture?', 'feature': 'F11', 'options': ['Thick', 'Thin']},
    {'question': 'What is your hair texture?', 'feature': 'F12', 'options': ['Thick', 'Thin']},
    {'question': 'Is your scalp hair prone to?', 'feature': 'F13', 'options': ['Graying', 'Falling', 'Breaking', 'Split at ends', 'Both', 'None']},
    {'question': 'What is your hair nature?', 'feature': 'F14', 'options': ['Dry', 'Oily', 'Normal', 'Seasonal']},
    {'question': 'Do you have brittle nails?', 'feature': 'F15', 'options': ['Yes', 'No']},
    {'question': 'Do you have cracked palm?', 'feature': 'F16', 'options': ['Yes', 'No']},
    {'question': 'Do you have cracked sole?', 'feature': 'F17', 'options': ['Yes', 'No']},
    {'question': 'Do you have cracked lips?', 'feature': 'F18', 'options': ['Yes', 'No']},
    {'question': 'How is your appetite (regularity)?', 'feature': 'F19', 'options': ['Regular', 'Irregular']},
    {'question': 'How is your appetite (frequency)?', 'feature': 'F20', 'options': ['Frequent', 'Infrequent']},
    {'question': 'Do you like sweet taste?', 'feature': 'F21', 'options': ['Like_Sweet', 'Donotlike_Sweet']},
    {'question': 'Do you like sour taste?', 'feature': 'F22', 'options': ['Like_Sour', 'Donotlike_Sour']},
    {'question': 'Do you like salty taste?', 'feature': 'F23', 'options': ['Like_Salty', 'Donotlike_Salty']},
    {'question': 'Do you like bitter taste?', 'feature': 'F24', 'options': ['Like_Bitter', 'Donotlike_Bitter']},
    {'question': 'Do you like pungent taste?', 'feature': 'F25', 'options': ['Like_Pungent', 'Donotlike_Pungent']},
    {'question': 'Do you like astringent taste?', 'feature': 'F26', 'options': ['Like_Astringent', 'Donotlike_Astringent']},
    {'question': 'What type of food/beverages do you prefer?', 'feature': 'F27', 'options': ['Cold', 'Warm', 'Any', 'None']},
    {'question': 'How much food can you consume when hungry?', 'feature': 'F28', 'options': ['Low', 'Medium', 'High', 'Variable']},
    {'question': 'Are you able to digest the food consumed?', 'feature': 'F29', 'options': ['Always yes', 'If excess is taken causes indigestion otherwise yes', 'Always with difficulty', 'Cannot say']},
    {'question': 'Do you prefer food rich in fats?', 'feature': 'F30', 'options': ['Butter', 'Ghee', 'Cheese', 'Animal Fat', 'Oil or oily articles', 'None']},
    {'question': 'Does your body temperature generally remain?', 'feature': 'F31', 'options': ['Higher compared to others', 'Lower compared to others', 'Average', 'Variable']},
    {'question': 'How about your perspiration?', 'feature': 'F32', 'options': ['Profuse', 'Moderate', 'Less', 'Variable']},
    {'question': 'How about your sleep (amount)?', 'feature': 'F33', 'options': ['Less sleep (<6 hrs)', 'Moderate sleep (6-8hrs)', 'Heavy sleep (>8hrs)', 'Variable']},
    {'question': 'Do you get sleep immediately after going to bed?', 'feature': 'F34', 'options': ['Yes', 'After few minutes / doing reading etc.', 'No it takes long time to fall asleep']},
    {'question': 'What is the quality of your sleep?', 'feature': 'F35', 'options': ['Deep', 'Moderate/Sound', 'Shallow']},
    {'question': 'How about your bowel habits?', 'feature': 'F36', 'options': ['Regular', 'Irregular', 'Occasionally Irregular']},
    {'question': 'Do you tend to have?', 'feature': 'F37', 'options': ['Constipation', 'Loose motions', 'None']},
    {'question': 'What is your stool consistency?', 'feature': 'F38', 'options': ['Hard', 'Loose', 'Soft', 'Semisolid', 'Medium']},
    {'question': 'How about changes in your body weight?', 'feature': 'F39', 'options': ['Gain weight easily and loose easily', 'Difficulty in gaining weight', 'Gain weight easily but loose with difficulty', 'Stable']},
    {'question': 'Do you have body odor?', 'feature': 'F40', 'options': ['Strong', 'Mild', 'Very Mild']},
    {'question': 'Which weather do you prefer?', 'feature': 'F41', 'options': ['Cold', 'Warm', 'Both Cold & Warm', 'Seasonal Transition', 'All', 'None']},
    {'question': 'In which weather do you have health problems?', 'feature': 'F42', 'options': ['Cold', 'Warm', 'Both Cold & Warm', 'Seasonal Transition', 'All', 'None']},
    {'question': 'How frequently do you fall ill?', 'feature': 'F43', 'options': ['Frequently', 'Rarely', 'Moderately']},
    {'question': 'If you fall ill, do you get cured easily?', 'feature': 'F44', 'options': ['Yes, mostly on its own', 'No, it takes long time & effort to get cured', 'Moderate efforts needed like diet, rest & medicine']},
    {'question': 'What is the amount of your speaking?', 'feature': 'F45', 'options': ['Excessive', 'Less', 'Moderate']},
    {'question': 'What is the quality of your voice?', 'feature': 'F46', 'options': ['Low', 'Feeble', 'Weak', 'Broken', 'Rough', 'Deep', 'Good toned', 'Sharp', 'Clear', 'High pitched', 'Loud', 'Soft, Pleasing']},
    {'question': 'What is the speed/style of your speaking?', 'feature': 'F47', 'options': ['Slow', 'Quick', 'Medium', 'Variably']},
    {'question': 'What is the level of your hand movement?', 'feature': 'F48', 'options': ['High/Excessive', 'Less', 'Moderate']},
    {'question': 'What is the level of your leg movement?', 'feature': 'F49', 'options': ['High/Excessive', 'Less', 'Moderate']},
    {'question': 'What is the level of your eyebrow movement?', 'feature': 'F50', 'options': ['High/Excessive', 'Less', 'Moderate']},
    {'question': 'What is the level of your shoulder movement?', 'feature': 'F51', 'options': ['High/Excessive', 'Less', 'Moderate']},
    {'question': 'What is the level of your overall movement?', 'feature': 'F52', 'options': ['High/Excessive', 'Less', 'Moderate']},
    {'question': 'What is your mental strength?', 'feature': 'F53', 'options': ['Get stressed / disturbed frequently and can be counselled by others', 'Get stressed / disturbed easily and overcome it by own / with some time', 'Get stressed with difficulty and can overcome on own', 'Get stressed easily and cannot be counseled easily by others']},
    {'question': 'How frequently do you feel tired?', 'feature': 'F54', 'options': ['On During routine work', 'After doing extra work/ heavy work', 'Not even after heavy work']},
    {'question': 'How quickly can you memorize things?', 'feature': 'F55', 'options': ['Moderately', 'Quickly', 'Slowly', 'Variably']},
    {'question': 'How forgetful are you?', 'feature': 'F56', 'options': ['Quickly', 'Moderately', 'Slowly', 'Variably']},
    {'question': 'How is your memory retention power?', 'feature': 'F57', 'options': ['Good', 'Medium', 'Poor', 'Variable']},
    {'question': 'Are you a...', 'feature': 'F58', 'options': ['Regular and routine observer', 'Spontaneous and moderate routine observer', 'Loving to experiment with routines and change them very readily']},
    {'question': 'Do you like to...', 'feature': 'F59', 'options': ['Move around and interact with people and explore', 'Be seated and keep confined to own work', 'Move around moderately and not sit for very long hours']},
]

# Streamlit app
st.title("Prakriti Assessment Tool")
st.write("Answer the 59 questions below to find your Vata, Pitta, and Kapha percentages. Select an option for each question and click 'Submit'.")

# Collect responses
responses = {}
for q in questionnaire:
    responses[q['feature']] = st.selectbox(q['question'], q['options'], key=q['feature'])

# Submit button
if st.button("Submit"):
    # Fill remaining features (F60-F132) with mode values
    for feature in features.columns:
        if feature not in responses:
            responses[feature] = features[feature].mode()[0]
    
    # Predict Prakriti
    new_df = pd.DataFrame([responses])
    for column in new_df.columns:
        if column in label_encoders:
            try:
                new_df[column] = label_encoders[column].transform(new_df[column].astype(str))
            except ValueError:
                new_df[column] = label_encoders[column].classes_[0]
    
    for feature in features.columns:
        if feature not in new_df.columns:
            new_df[feature] = features[feature].mode()[0]
    
    probs = model.predict_proba(new_df)[0]
    prediction = {
        'Vata': probs[le_label.transform(['Vata'])[0]] * 100,
        'Pitta': probs[le_label.transform(['Pitta'])[0]] * 100,
        'Kapha': probs[le_label.transform(['Kapha'])[0]] * 100
    }
    
    # Display results
    st.subheader("Your Prakriti Assessment Results")
    st.write(f"**Vata**: {prediction['Vata']:.2f}%")
    st.write(f"**Pitta**: {prediction['Pitta']:.2f}%")
    st.write(f"**Kapha**: {prediction['Kapha']:.2f}%")
