from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

# Route for index page
@application.route('/')
def index():
    return render_template('index.html')

# Route for prediction page
@application.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Create custom data object
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )

            # Convert to DataFrame
            pred_df = data.get_data_as_data_frame()
            print("Input DataFrame:\n", pred_df)
            print("Before Prediction")

            # Run prediction pipeline
            predict_pipeline = PredictPipeline()
            print("Running Prediction")
            results = predict_pipeline.predict(pred_df)
            print("Prediction Result:", results)

            # Return result to template
            return render_template('home.html', results=round(results[0], 2))

        except Exception as e:
            print("Error during prediction:", e)
            return render_template('home.html', results="Error in prediction. Please check your inputs.")

if __name__ == "__main__":
    application.run(host="0.0.0.0", debug=True, port=5000)
