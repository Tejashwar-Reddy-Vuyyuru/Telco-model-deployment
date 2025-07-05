from flask import Flask,render_template,request
import pandas as pd
import joblib

app = Flask(__name__)
telco_stats = joblib.load('/workspaces/Telco-model-deployment/model/telco_st.joblib')

model = telco_stats['model']
input_cols = telco_stats['input_cols']
target_col = telco_stats['target_col']
encode_cols = telco_stats['encode_cols']
numerical_cols = telco_stats['numerical_cols']

@app.route('/',methods = ['GET','POST'])
def index():
    prediction = None
    if request.method == 'POST':
        form_data = {
            'Zip_Code': request.form['Zip_Code'],
            'Latitude': float(request.form['Latitude']),
            'Longitude': float(request.form['Longitude']),
            'Tenure_Months': float(request.form['Tenure_Months']),
            'Monthly_Charges':float(request.form['Monthly_Charges']),
            'Total_Charges':float(request.form['Total_Charges']),
            'City': request.form['City'],
            'Gender': request.form['Gender'],
            'Senior_Citizen': request.form['Senior_Citizen'],
            'Partner': request.form['Partner'],
            'Dependents': request.form['Dependents'],
            'Phone_Service': request.form['Phone_Service'],
            'Multiple_Lines': request.form['Multiple_Lines'],
            'Internet_Service': request.form['Internet_Service'],
            'Online_Security': request.form['Online_Security'],
            'Online_Backup': request.form['Online_Backup'],
            'Device_Protection': request.form['Device_Protection'],
            'Tech_Support': request.form['Tech_Support'],
            'Streaming_TV': request.form['Streaming_TV'],
            'Streaming_Movies': request.form['Streaming_Movies'],
            'Contract': request.form['Contract'],
            'Paperless_Billing': request.form['Paperless_Billing'],
            'Payment_Method': request.form['Payment_Method'],
            
        }

        input_df = pd.DataFrame([form_data])
        
        # Convert numeric columns
        for col in numerical_cols:
            input_df[col] = pd.to_numeric(input_df[col])
        
        # One-hot encode categorical variables
        input_df_encoded = pd.get_dummies(input_df, columns=encode_cols).astype(int) 
        
        # Ensure we have all expected columns (from training)
        # Add missing dummy columns with 0 values
        for col in input_cols:
            if col not in input_df_encoded.columns:
                input_df_encoded[col] = 0
        
        # Reorder columns to match training data
        input_df_encoded = input_df_encoded.reindex(columns=input_cols, fill_value=0)

        # Make prediction
        prediction_proba = model.predict_proba(input_df_encoded)
        prediction = 'Yes' if model.predict(input_df_encoded)[0] == 1 else 'No'
        confidence = round(prediction_proba[0][1] * 100, 2) if prediction == 'Yes' else round(prediction_proba[0][0] * 100, 2)
        
        return render_template('index.html', 
                            prediction=prediction, 
                            confidence=confidence,
                            form_data=form_data)
    
    return render_template('index.html', prediction=None)

    
if __name__ == '__main__':
    app.run(debug = True)
