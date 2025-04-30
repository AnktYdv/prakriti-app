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
    le = LabelEncoder()
    features[column] = le.fit_transform(features[column].astype(str))
    label_encoders[column] = le

# Encode labels
le_label = LabelEncoder()
y_encoded = le_label.fit_transform(labels)

# Train final model
model = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', random_state=42)
model.fit(features, y_encoded)

# Define questionnaire with 132 features
questionnaire = [
    {'question': 'What is your gender?', 'feature': 'GENDER', 'options': ['Male', 'Female']},
    {'question': 'What is your body frame?', 'feature': 'F1', 'options': ['Narrow', 'Medium', 'Wide']},
    {'question': 'What is your body build (bulk)?', 'feature': 'F2', 'options': ['Weakly developed', 'Moderately developed', 'Well developed']},
    {'question': 'What is your body build (musculature)?', 'feature': 'F3', 'options': ['Thin Musculature', 'Soft and Loosely knitted Musculature', 'Smooth and Firmly knitted Musculature']},
    {'question': 'What is your forehead length?', 'feature': 'F4', 'options': ['Small', 'Medium', 'Large']},
    {'question': 'What is your nails texture?', 'feature': 'F5', 'options': ['Smooth', 'Soft', 'Rough']},
    {'question': 'What is your nails colour?', 'feature': 'F6', 'options': ['Reddish', 'Pale', 'Pink']},
    {'question': 'What is your finger nail size?', 'feature': 'F7', 'options': ['Small', 'Medium', 'Large']},
    {'question': 'Does your skin appear cracked?', 'feature': 'F8', 'options': ['Yes', 'No']},
    {'question': 'Does your skin appear lustrous?', 'feature': 'F9', 'options': ['Yes', 'No']},
    {'question': 'Does your skin appear rough?', 'feature': 'F10', 'options': ['Yes', 'No']},
    {'question': 'Does your skin have moles?', 'feature': 'F11', 'options': ['Yes', 'No']},
    {'question': 'Does your skin have marks?', 'feature': 'F12', 'options': ['Yes', 'No']},
    {'question': 'Does your skin have pimples?', 'feature': 'F13', 'options': ['Yes', 'No']},
    {'question': 'Does your skin have freckles?', 'feature': 'F14', 'options': ['Yes', 'No']},
    {'question': 'Does your skin have wrinkles?', 'feature': 'F15', 'options': ['Yes', 'No']},
    {'question': 'Does your skin have none of the above?', 'feature': 'F16', 'options': ['Yes', 'No']},
    {'question': 'What is your skin colour/complexion?', 'feature': 'F17', 'options': ['Fair with reddish tinge', 'Fair with yellowish tinge', 'Dark', 'Dusky', 'Wheatish', 'Fair with pale tinge', 'Fair with pink tinge']},
    {'question': 'What is your skin nature?', 'feature': 'F18', 'options': ['Dry', 'Oily', 'Normal', 'Seasonal']},
    {'question': 'What is your skin texture?', 'feature': 'F19', 'options': ['Thick', 'Thin']},
    {'question': 'What is your hair texture?', 'feature': 'F20', 'options': ['Thick', 'Thin']},
    {'question': 'Is your scalp hair prone to graying?', 'feature': 'F21', 'options': ['Yes', 'No']},
    {'question': 'Is your scalp hair prone to falling?', 'feature': 'F22', 'options': ['Yes', 'No']},
    {'question': 'Is your scalp hair prone to breaking?', 'feature': 'F23', 'options': ['Yes', 'No']},
    {'question': 'Is your scalp hair prone to split ends?', 'feature': 'F24', 'options': ['Yes', 'No']},
    {'question': 'Is your scalp hair prone to both graying and falling?', 'feature': 'F25', 'options': ['Yes', 'No']},
    {'question': 'Is your scalp hair prone to none of the above?', 'feature': 'F26', 'options': ['Yes', 'No']},
    {'question': 'What is your hair nature?', 'feature': 'F27', 'options': ['Dry', 'Oily', 'Normal', 'Seasonal']},
    {'question': 'Do you have brittle nails?', 'feature': 'F28', 'options': ['Yes', 'No']},
    {'question': 'Do you have cracked palms?', 'feature': 'F29', 'options': ['Yes', 'No']},
    {'question': 'Do you have cracked soles?', 'feature': 'F30', 'options': ['Yes', 'No']},
    {'question': 'Do you have cracked lips?', 'feature': 'F31', 'options': ['Yes', 'No']},
    {'question': 'How is your appetite (regularity)?', 'feature': 'F32', 'options': ['Regular', 'Irregular']},
    {'question': 'How is your appetite (frequency)?', 'feature': 'F33', 'options': ['Frequent', 'Infrequent']},
    {'question': 'Do you like sweet taste?', 'feature': 'F34', 'options': ['Yes', 'No']},
    {'question': 'Do you like sour taste?', 'feature': 'F35', 'options': ['Yes', 'No']},
    {'question': 'Do you like salty taste?', 'feature': 'F36', 'options': ['Yes', 'No']},
    {'question': 'Do you like bitter taste?', 'feature': 'F37', 'options': ['Yes', 'No']},
    {'question': 'Do you like pungent taste?', 'feature': 'F38', 'options': ['Yes', 'No']},
    {'question': 'Do you like astringent taste?', 'feature': 'F39', 'options': ['Yes', 'No']},
    {'question': 'What type of food/beverages do you prefer?', 'feature': 'F40', 'options': ['Cold', 'Warm', 'Any', 'None']},
    {'question': 'How much food can you consume when hungry?', 'feature': 'F41', 'options': ['Low', 'Medium', 'High', 'Variable']},
    {'question': 'Are you able to digest the food consumed?', 'feature': 'F42', 'options': ['Always yes', 'If excess is taken causes indigestion otherwise yes', 'Always with difficulty', 'Cannot say']},
    {'question': 'Do you prefer butter?', 'feature': 'F43', 'options': ['Yes', 'No']},
    {'question': 'Do you prefer ghee?', 'feature': 'F44', 'options': ['Yes', 'No']},
    {'question': 'Do you prefer cheese?', 'feature': 'F45', 'options': ['Yes', 'No']},
    {'question': 'Do you prefer animal fat?', 'feature': 'F46', 'options': ['Yes', 'No']},
    {'question': 'Do you prefer oil or oily articles?', 'feature': 'F47', 'options': ['Yes', 'No']},
    {'question': 'Do you prefer none of the above fats?', 'feature': 'F48', 'options': ['Yes', 'No']},
    {'question': 'Does your body temperature generally remain?', 'feature': 'F49', 'options': ['Higher compared to others', 'Lower compared to others', 'Average', 'Variable']},
    {'question': 'How about your perspiration?', 'feature': 'F50', 'options': ['Profuse', 'Moderate', 'Less', 'Variable']},
    {'question': 'How about your sleep (amount)?', 'feature': 'F51', 'options': ['Less sleep (<6 hrs)', 'Moderate sleep (6-8hrs)', 'Heavy sleep (>8hrs)', 'Variable']},
    {'question': 'Do you get sleep immediately after going to bed?', 'feature': 'F52', 'options': ['Yes', 'After few minutes / doing reading etc.', 'No it takes long time to fall asleep']},
    {'question': 'What is the quality of your sleep?', 'feature': 'F53', 'options': ['Deep', 'Moderate/Sound', 'Shallow']},
    {'question': 'How about your bowel habits?', 'feature': 'F54', 'options': ['Regular', 'Irregular', 'Occasionally Irregular']},
    {'question': 'Do you tend to have constipation?', 'feature': 'F55', 'options': ['Yes', 'No']},
    {'question': 'Do you tend to have loose motions?', 'feature': 'F56', 'options': ['Yes', 'No']},
    {'question': 'Do you tend to have normal bowels?', 'feature': 'F57', 'options': ['Yes', 'No']},
    {'question': 'What is your stool consistency?', 'feature': 'F58', 'options': ['Hard', 'Loose', 'Soft', 'Semisolid', 'Medium']},
    {'question': 'How about changes in your body weight?', 'feature': 'F59', 'options': ['Gain weight easily and loose easily', 'Difficulty in gaining weight', 'Gain weight easily but loose with difficulty', 'Stable']},
    {'question': 'Do you have body odor?', 'feature': 'F60', 'options': ['Strong', 'Mild', 'Very Mild']},
    {'question': 'Do you prefer cold weather?', 'feature': 'F61', 'options': ['Yes', 'No']},
    {'question': 'Do you prefer warm weather?', 'feature': 'F62', 'options': ['Yes', 'No']},
    {'question': 'Do you prefer both cold and warm weather?', 'feature': 'F63', 'options': ['Yes', 'No']},
    {'question': 'Do you prefer seasonal transition weather?', 'feature': 'F64', 'options': ['Yes', 'No']},
    {'question': 'Do you prefer all weather?', 'feature': 'F65', 'options': ['Yes', 'No']},
    {'question': 'Do you prefer none of the weather?', 'feature': 'F66', 'options': ['Yes', 'No']},
    {'question': 'Do you have health problems in cold weather?', 'feature': 'F67', 'options': ['Yes', 'No']},
    {'question': 'Do you have health problems in warm weather?', 'feature': 'F68', 'options': ['Yes', 'No']},
    {'question': 'Do you have health problems in both cold and warm weather?', 'feature': 'F69', 'options': ['Yes', 'No']},
    {'question': 'Do you have health problems in seasonal transition?', 'feature': 'F70', 'options': ['Yes', 'No']},
    {'question': 'Do you have health problems in all weather?', 'feature': 'F71', 'options': ['Yes', 'No']},
    {'question': 'Do you have no weather-related health problems?', 'feature': 'F72', 'options': ['Yes', 'No']},
    {'question': 'How frequently do you fall ill?', 'feature': 'F73', 'options': ['Frequently', 'Rarely', 'Moderately']},
    {'question': 'If you fall ill, do you get cured easily?', 'feature': 'F74', 'options': ['Yes, mostly on its own', 'No, it takes long time & effort to get cured', 'Moderate efforts needed like diet, rest & medicine']},
    {'question': 'What is the amount of your speaking?', 'feature': 'F75', 'options': ['Excessive', 'Less', 'Moderate']},
    {'question': 'Is your voice low?', 'feature': 'F76', 'options': ['Yes', 'No']},
    {'question': 'Is your voice feeble?', 'feature': 'F77', 'options': ['Yes', 'No']},
    {'question': 'Is your voice weak?', 'feature': 'F78', 'options': ['Yes', 'No']},
    {'question': 'Is your voice broken?', 'feature': 'F79', 'options': ['Yes', 'No']},
    {'question': 'Is your voice rough?', 'feature': 'F80', 'options': ['Yes', 'No']},
    {'question': 'Is your voice deep?', 'feature': 'F81', 'options': ['Yes', 'No']},
    {'question': 'What is the speed/style of your speaking?', 'feature': 'F82', 'options': ['Slow', 'Quick', 'Medium', 'Variably']},
    {'question': 'What is the level of your hand movement?', 'feature': 'F83', 'options': ['High/Excessive', 'Less', 'Moderate']},
    {'question': 'What is the level of your leg movement?', 'feature': 'F84', 'options': ['High/Excessive', 'Less', 'Moderate']},
    {'question': 'What is the level of your eyebrow movement?', 'feature': 'F85', 'options': ['High/Excessive', 'Less', 'Moderate']},
    {'question': 'What is the level of your shoulder movement?', 'feature': 'F86', 'options': ['High/Excessive', 'Less', 'Moderate']},
    {'question': 'What is the level of your overall movement?', 'feature': 'F87', 'options': ['High/Excessive', 'Less', 'Moderate']},
    {'question': 'What is your mental strength?', 'feature': 'F88', 'options': ['Get stressed / disturbed frequently and can be counselled by others', 'Get stressed / disturbed easily and overcome it by own / with some time', 'Get stressed with difficulty and can overcome on own', 'Get stressed easily and cannot be counseled easily by others']},
    {'question': 'How frequently do you feel tired?', 'feature': 'F89', 'options': ['On During routine work', 'After doing extra work/ heavy work', 'Not even after heavy work']},
    {'question': 'How quickly can you memorize things?', 'feature': 'F90', 'options': ['Moderately', 'Quickly', 'Slowly', 'Variably']},
    {'question': 'How forgetful are you?', 'feature': 'F91', 'options': ['Quickly', 'Moderately', 'Slowly', 'Variably']},
    {'question': 'How is your memory retention power?', 'feature': 'F92', 'options': ['Good', 'Medium', 'Poor', 'Variable']},
    {'question': 'Are you a...', 'feature': 'F93', 'options': ['Regular and routine observer', 'Spontaneous and moderate routine observer', 'Loving to experiment with routines and change them very readily']},
    {'question': 'Do you like to...', 'feature': 'F94', 'options': ['Move around and interact with people and explore', 'Be seated and keep confined to own work', 'Move around moderately and not sit for very long hours']},
]

# Streamlit app
st.title("Prakriti Assessment Tool")
st.write("Answer the questions below to find your Vata, Pitta, and Kapha percentages. Select an option for each question and click 'Submit'.")

# Collect responses
responses = {}
for q in questionnaire:
    responses[q['feature']] = st.selectbox(q['question'], q['options'], key=q['feature'])

# Submit button
if st.button("Submit"):
    # Create DataFrame
    new_df = pd.DataFrame([responses])
    
    # Encode responses
    for column in new_df.columns:
        if column in label_encoders:
            try:
                new_df[column] = new_df[column].apply(lambda x: label_encoders[column].transform([str(x)])[0] if x in label_encoders[column].classes_ else label_encoders[column].classes_[0])
            except Exception as e:
                st.error(f"Error encoding column {column}: {str(e)}")
                st.stop()
        else:
            new_df[column] = pd.to_numeric(new_df[column], errors='coerce').fillna(0)

    # Ensure all features are present
    for feature in features.columns:
        if feature not in new_df.columns:
            mode_value = data_features[feature].mode()[0]
            if feature in label_encoders:
                try:
                    new_df[feature] = label_encoders[feature].transform([str(mode_value)])[0]
                except:
                    new_df[feature] = label_encoders[feature].classes_[0]
            else:
                new_df[feature] = mode_value

    # Verify numeric types
    if not new_df.select_dtypes(include=[np.number]).columns.tolist() == new_df.columns.tolist():
        st.error("Error: Some features are not numeric. Please check your inputs.")
        st.stop()

    try:
        # Predict
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
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
