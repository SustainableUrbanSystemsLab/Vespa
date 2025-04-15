# 25-SP-EnergyPlugin

**25-SP-EnergyPlugin** is a Rhino plugin designed to support architects in making energy-informed design decisions early in the building process. The plugin enables users to create or modify building models and receive predictions for heating and cooling loads using a machine learning (ML) model.

## Features

- **Building Creation & Editing**  
  Users can:
  - Create buildings from scratch within the Rhino environment.
  - Use a default pre-loaded building model as a starting point.
  - Edit any building geometry directly in Rhino.

- **Custom ML Model Support**  
  Users may:
  - Load their own machine learning model file in `.joblib` format.
  - Use the plugin's default pre-trained ML model for energy prediction.

- **Automatic Feature Extraction**  
  - The plugin includes a feature extraction module that automatically reads the `.joblib` file and identifies required input features for prediction.
  - Upon model upload, the `.joblib` file is converted into the ONNX format for compatibility with C# in the Rhino SDK environment.
  - To support automated prediction, we extract relevant building parameters such as:
    - **Building height** (calculated as the difference between maximum and minimum points along the Y-axis),
    - **Number of stories** (estimated either by dividing height by a typical floor height or by counting slab elements),
    - **Wall area**, **roof area**, and **window area** (computed by converting surfaces to Breps, deconstructing them into faces, and summing the dominant faces from each Brep).
  - These calculations were initially implemented using Grasshopper components. All building elements must be assigned to specific layers—`Wall`, `Slab`, `Window`, and `Roof`—for accurate extraction.
  - The process was later translated into Python to streamline and automate these computations directly within the plugin workflow, improving both efficiency and consistency.

- **Energy Load Prediction**  
  After the building design is finalized and features are automatically extracted:
  - The feature vector - containing values like building height, number of stories, wall area, roof area, and window area - is passed into the selected ML model (either user-provided or the default).
  - The model processes these features and outputs predicted values for heating and cooling loads.
  - These predictions help inform design decisions by providing quick, data-driven feedback on the energy performance of the current building configuration.

## Workflow Overview

1. Open Rhino and launch the **25-SP-EnergyPlugin** panel.
2. Choose to:
   - Create a new building from scratch, or
   - Load and edit the default building model.
3. (Optional) Upload a custom `.joblib` ML model for predictions.
4. The plugin:
   - Extracts required features from the ML model.
   - Converts the model to ONNX for inference.
5. Once the building is finalized:
   - Extracts the building features.
   - Runs inference to predict heating and cooling loads.
6. View and use the energy predictions to inform further design iterations.

## Requirements

- **Rhino 7 or later**
- **Windows OS** – Required for Rhino and .NET integration
- **.NET Framework 4.8+** – Compatible with Rhino SDK and ONNX Runtime
- **Python 3.8+** – Used for model training, feature extraction, and `.joblib` to `.onnx` conversion
- **scikit-learn ML model (optional)** – If using a custom model, it must be:
  - Serialized in `.joblib` format
  - Trained with features compatible with the plugin's extraction schema (e.g., height, stories, wall/roof/window area)

## Tech Stack

- **Rhino SDK (C#)** – Core plugin and geometry interaction
- **Grasshopper (C#)** – Dynamic UI for model input and inference
- **ONNX Runtime** – High-performance inference in the C# environment
- **Python** – Backend for model training, preprocessing, and feature extraction
- **Joblib** – Serialization for scikit-learn models
- **sklearn-onnx** – Converts `.joblib` models to `.onnx` format for C# compatibility
- **NumPy / pandas** – Data processing tools used during feature pipeline preparation
