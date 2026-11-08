import os
import json
import joblib
import numpy as np
import pandas as pd
import gradio as gr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load saved models
mass_model = joblib.load(
    os.path.join(BASE_DIR, "stellar_mass_model.pkl")
)

sfr_model = joblib.load(
    os.path.join(BASE_DIR, "sfr_model.pkl")
)

with open(
    os.path.join(BASE_DIR, "model_metadata.json"),
    "r"
) as f:
    metadata = json.load(f)

features = metadata["input_features"]

# Load real demonstration galaxy
with open(
    os.path.join(BASE_DIR, "sample_input.json"),
    "r"
) as f:
    sample_input = json.load(f)


def predict_galaxy(*values):
    try:
        if len(values) != len(features):
            raise ValueError("Incorrect number of input features.")

        numeric_values = [float(v) for v in values]

        if not np.isfinite(numeric_values).all():
            raise ValueError("Enter valid finite numerical values.")

        input_df = pd.DataFrame(
            [numeric_values],
            columns=features
        )

        mass_prediction = float(
            mass_model.predict(input_df)[0]
        )

        sfr_prediction = float(
            sfr_model.predict(input_df)[0]
        )

        return (
            f"{mass_prediction:.6f}",
            f"{sfr_prediction:.6f}",
            "SUCCESS - Both galaxy properties predicted successfully."
        )

    except Exception as error:
        return (
            "Not available",
            "Not available",
            f"Prediction failed: {str(error)}"
        )


with gr.Blocks(
    title="Galaxy Evolution Predictor"
) as demo:

    gr.Markdown("""
    # Galaxy Evolution Predictor

    ### AI-powered prediction of galaxy properties

    Predict stellar mass and star formation rate using
    nine-band GAaP photometry, measurement errors and
    photometric redshift.

    **Prediction Engine:** HistGradientBoostingRegressor
    """)

    gr.Markdown(
        "Inputs are pre-filled using a real galaxy from "
        "the KiDS DR4 catalogue. You can modify the values."
    )

    inputs = []

    with gr.Row():

        with gr.Column():
            gr.Markdown("### Photometric Magnitudes")

            for feature in features[:9]:
                inputs.append(
                    gr.Number(
                        label=feature,
                        value=float(sample_input[feature])
                    )
                )

        with gr.Column():
            gr.Markdown("### Measurement Errors and Redshift")

            for feature in features[9:]:
                inputs.append(
                    gr.Number(
                        label=feature,
                        value=float(sample_input[feature])
                    )
                )

    gr.Markdown("## Generate Predictions")

    predict_button = gr.Button(
        "PREDICT GALAXY PROPERTIES",
        variant="primary",
        size="lg"
    )

    with gr.Row():
        mass_output = gr.Textbox(
            label="Predicted Stellar Mass",
            interactive=False
        )

        sfr_output = gr.Textbox(
            label="Predicted Star Formation Rate",
            interactive=False
        )

    status_output = gr.Textbox(
        label="Prediction Status",
        interactive=False
    )

    predict_button.click(
        fn=predict_galaxy,
        inputs=inputs,
        outputs=[
            mass_output,
            sfr_output,
            status_output
        ]
    )

    gr.Markdown("""
    ---

    ### About this model

    - Input: 19 photometric and redshift features
    - Output 1: MASS_MED_GaZNet prediction
    - Output 2: SFR_MED_GaZNet prediction
    - Algorithms: Two separately trained
      HistGradientBoostingRegressor models

    **Scientific note:** Predictions reproduce catalogue-derived
    estimates, not independent physical measurements.
    Values use the target columns' native numerical units.
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
