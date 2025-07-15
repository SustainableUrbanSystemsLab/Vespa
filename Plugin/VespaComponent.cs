using Grasshopper;
using Grasshopper.Kernel;
using Rhino.Geometry;
using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using System.IO;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Linq;

namespace VespaCore
{
    // This component implements IGH_VariableParameterComponent so that its input parameters can be changed dynamically.
    public class VespaComponent : GH_Component, IGH_VariableParameterComponent
    {
        // Store the current model file path.
        private string currentModelPath = "";

        private bool dynamicInputsUpdated = false;

        // This list holds the names of all dynamically created parameters.
        private List<string> modelInputNames = new List<string>();

        // Maps each ONNX input key (for example, "input") to a list of dynamic parameter names.
        private Dictionary<string, List<string>> inputMapping = new Dictionary<string, List<string>>();

        public VespaComponent()
          : base("Predict", "Predict",
                "Predict building energy load using an ONNX model with dynamic inputs",
                "Vespa", "Inference")
        {
        }

        public override GH_Exposure Exposure => GH_Exposure.primary;

        // Initially, only the Model file path is registered.
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddTextParameter("Model", "M", "Path to ONNX model file", GH_ParamAccess.item);
            // Dynamic inputs will be added later based on the model metadata.
        }

        // Output 0: Model output metadata; Output 1: Model prediction.
        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddTextParameter("Output MetaData", "OM", "Model output metadata", GH_ParamAccess.list);
            pManager.AddNumberParameter("Output", "O", "Predicted energy load", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            // 1. Get the model file path.
            string modelPath = "";
            if (!DA.GetData(0, ref modelPath))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Model file path not provided.");
                return;
            }
            if (string.IsNullOrEmpty(modelPath) || !File.Exists(modelPath))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Model not found.");
                return;
            }

            // 2. If the model has changed, mark that dynamic inputs need updating.
            if (modelPath != currentModelPath)
            {
                currentModelPath = modelPath;
                dynamicInputsUpdated = false;
            }

            // 3. If dynamic inputs are not yet updated, open the model, read its metadata, and update inputs.
            if (!dynamicInputsUpdated)
            {
                try
                {
                    using (var session = new InferenceSession(modelPath))
                    {
                        UpdateDynamicInputs(session);
                    }
                    dynamicInputsUpdated = true;
                    // Expire the solution so the component re-solves with the new dynamic inputs.
                    ExpireSolution(true);
                    return;
                }
                catch (Exception ex)
                {
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Error, $"Error loading ONNX model: {ex.Message}");
                    return;
                }
            }

            // 4. Gather dynamic input values.
            // The inputs are grouped by their original ONNX input key.
            var assembledInputs = new Dictionary<string, float[]>();
            foreach (var kvp in inputMapping)
            {
                string origKey = kvp.Key;
                List<string> dynNames = kvp.Value;
                float[] values = new float[dynNames.Count];
                for (int j = 0; j < dynNames.Count; j++)
                {
                    // Find the parameter index by matching the dynamic parameter's name.
                    int index = -1;
                    for (int k = 1; k < Params.Input.Count; k++) // index 0 is the Model input.
                    {
                        if (Params.Input[k].Name == dynNames[j])
                        {
                            index = k;
                            break;
                        }
                    }
                    if (index == -1)
                    {
                        AddRuntimeMessage(GH_RuntimeMessageLevel.Error, $"Dynamic parameter {dynNames[j]} not found.");
                        return;
                    }
                    double val = 0;
                    if (!DA.GetData(index, ref val))
                    {
                        AddRuntimeMessage(GH_RuntimeMessageLevel.Error, $"Missing value for parameter: {dynNames[j]}");
                        return;
                    }
                    values[j] = (float)val;
                }
                assembledInputs[origKey] = values;
            }

            // 5. Run the ONNX model.
            try
            {
                using (var session = new InferenceSession(currentModelPath))
                {
                    var inputs = new List<NamedOnnxValue>();
                    // For each ONNX input key, create a tensor from the assembled scalar values.
                    foreach (var kvp in assembledInputs)
                    {
                        string origKey = kvp.Key;
                        float[] vals = kvp.Value;
                        // Use shape [1, n] if more than one value; otherwise shape [1].
                        int[] shape = (vals.Length > 1) ? new[] { 1, vals.Length } : new[] { 1 };
                        var tensor = new DenseTensor<float>(vals, shape);
                        // IMPORTANT: Use the original key as the tensor name.
                        inputs.Add(NamedOnnxValue.CreateFromTensor(origKey, tensor));
                    }
                    using (var results = session.Run(inputs))
                    {
                        // Assume the first output is the prediction.
                        var outputTensor = results.First().AsTensor<float>();
                        float prediction = outputTensor[0];
                        DA.SetData(1, prediction);

                        // Output the model's output metadata.
                        var outputMetaData = new List<string>();
                        foreach (var output in session.OutputMetadata)
                        {
                            outputMetaData.Add($"{output.Key}: {output.Value.ElementType} - {string.Join("x", output.Value.Dimensions)}");
                        }
                        DA.SetDataList(0, outputMetaData);
                    }
                }
            }
            catch (Exception ex)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, $"Error during model inference: {ex.Message}");
            }
        }

        // This method reads the ONNX model metadata and dynamically creates input parameters.
        private void UpdateDynamicInputs(InferenceSession session)
        {
            // Remove all inputs except the first one (the Model file input).
            while (Params.Input.Count > 1)
            {
                Params.UnregisterInputParameter(Params.Input[1], true);
            }
            modelInputNames.Clear();
            inputMapping.Clear();

            // Loop through each input defined in the ONNX model metadata.
            foreach (var kvp in session.InputMetadata)
            {
                string origKey = kvp.Key;
                var dims = kvp.Value.Dimensions;
                int featureCount = 1;
                // If the tensor shape is [1, n] and n > 1, then the model expects n features.
                if (dims.Length > 1 && dims[1] > 1)
                {
                    featureCount = dims[1];
                }
                List<string> dynamicNames = new List<string>();
                // Create a separate scalar parameter for each feature.
                for (int i = 0; i < featureCount; i++)
                {
                    string paramName = (featureCount > 1) ? $"{origKey}_{i}" : origKey;
                    var param = new Grasshopper.Kernel.Parameters.Param_Number();
                    param.Name = paramName;
                    param.NickName = paramName;
                    param.Access = GH_ParamAccess.item;
                    param.Optional = true;
                    param.Description = (featureCount > 1)
                        ? $"Input {i + 1} of {origKey} (Type: {kvp.Value.ElementType})"
                        : $"Input for {origKey} (Type: {kvp.Value.ElementType})";
                    Params.RegisterInputParam(param);
                    modelInputNames.Add(paramName);
                    dynamicNames.Add(paramName);
                }
                inputMapping[origKey] = dynamicNames;
            }
        }

        #region IGH_VariableParameterComponent Implementation

        public bool CanInsertParameter(GH_ParameterSide side, int index)
        {
            return side == GH_ParameterSide.Input;
        }

        public bool CanRemoveParameter(GH_ParameterSide side, int index)
        {
            // Prevent removal of the first parameter ("Model").
            return side == GH_ParameterSide.Input && index > 0;
        }

        public IGH_Param CreateParameter(GH_ParameterSide side, int index)
        {
            // Not used�dynamic parameters are created in UpdateDynamicInputs.
            return null;
        }

        public bool DestroyParameter(GH_ParameterSide side, int index)
        {
            return side == GH_ParameterSide.Input && index > 0;
        }

        public void VariableParameterMaintenance()
        {
            // No additional maintenance is needed.
        }

        #endregion IGH_VariableParameterComponent Implementation

        protected override System.Drawing.Bitmap Icon => null;
        public override Guid ComponentGuid => new Guid("84fd73f3-62e5-4245-948f-fd9f9d554eca");
    }
}