from flask import Flask, request, render_template
from src.Airbnb.pipelines.Prediction_Pipeline import CustomData, PredictPipeline
import numpy as np

app = Flask(__name__)

# Define the home route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Helper converters
            def to_int(name, default=0):
                try:
                    return int(request.form.get(name) or default)
                except Exception:
                    return default

            def to_float(name, default=0.0):
                try:
                    return float(request.form.get(name) or default)
                except Exception:
                    return default

            def to_bool_flag(name):
                return True if request.form.get(name) == '1' else False

            city_map = {
                'New York': 'NYC',
                'San Francisco': 'SF',
                'Washington, D.C.': 'DC',
                'Los Angeles': 'LA',
                'Chicago': 'Chicago',
                'Boston': 'Boston'
            }

            room_map = {
                'Entire Home/Apt': 'Entire home/apt',
                'Private Room': 'Private room',
                'Shared Room': 'Shared room'
            }

            cancel_map = {
                'Flexible': 'flexible',
                'Moderate': 'moderate',
                'Strict': 'strict',
                'Super strict': 'super_strict_30',
                'Advanced Super Strict': 'super_strict_60'
            }

            data = CustomData(
                property_type=request.form.get("property_type") or 'Other',
                room_type=room_map.get(request.form.get("room_type"), request.form.get("room_type")),
                amenities=to_int("amenities", 2),
                accommodates=to_int("accommodates", 1),
                bathrooms=to_float("bathrooms", 0.0),
                bed_type=request.form.get("bed_type") or 'Real Bed',
                cancellation_policy=cancel_map.get(request.form.get("cancellation_policy"), request.form.get("cancellation_policy")),
                cleaning_fee=to_bool_flag("cleaning_fee"),
                city=city_map.get(request.form.get("city"), request.form.get("city")),
                host_has_profile_pic=('t' if to_bool_flag("host_has_profile_pic") else 'f'),
                host_identity_verified=('t' if to_bool_flag("host_identity_verified") else 'f'),
                host_response_rate=to_int("host_response_rate", 0),
                instant_bookable=('t' if to_bool_flag("instant_bookable") else 'f'),
                latitude=to_float("latitude", 0.0),
                longitude=to_float("longitude", 0.0),
                number_of_reviews=to_int("number_of_reviews", 0),
                review_scores_rating=to_int("review_scores_rating", 0),
                bedrooms=to_int("bedrooms", 0),
                beds=to_int("beds", 0)
            )

            final_data = data.get_data_as_dataframe()

            # Make prediction
            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predict(final_data)
            # Model predicts log_price — convert back to price in US$
            try:
                price = float(np.exp(pred[0]))
                result = round(price, 2)
            except Exception:
                result = round(float(pred[0]), 2)

            return render_template("index.html", result=result)

        except Exception as e:
            # Handle exceptions gracefully
            error_message = f"Error during prediction: {str(e)}"
            return render_template("error.html", error_message=error_message)

    else:
        # Render the initial page
        return render_template("index.html")

# Execution begins
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
