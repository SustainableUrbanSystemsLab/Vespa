using Grasshopper;
using Grasshopper.Kernel;
using Rhino.Geometry;
using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using System.IO;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Linq;

namespace BuildingEnergyMLVIP
{
    // Implement IGH_VariableParameterComponent so that Grasshopper recognizes our dynamic inputs.
    public class SmartEnergyComponent : GH_Component, IGH_VariableParameterComponent
    {
        // Store the current model file path.
        private string currentModelPath = "";

        private bool dynamicInputsUpdated = false;

        // This list holds the names of all dynamically created parameters.
        private List<string> modelInputNames = new List<string>();

        // This dictionary maps each original model input key (e.g. "input") to a list of dynamic parameter names.
        private Dictionary<string, List<string>> inputMapping = new Dictionary<string, List<string>>();

        public SmartEnergyComponent()
          : base("SmartEnergyComponent", "MLVIP",
                "Predict building energy load using an ONNX model with dynamic inputs",
                "Params", "General")
        {
        }

        public override GH_Exposure Exposure => GH_Exposure.primary;

        // Initially only register the Model file path.
        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddTextParameter("Model", "M", "Path to ONNX model file", GH_ParamAccess.item);
            // The dynamic inputs will be added later based on the model metadata.
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddTextParameter("Output MetaData", "OM", "Model output metadata", GH_ParamAccess.list);
            pManager.AddNumberParameter("Output", "O", "Predicted energy load", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            // 1. Get the model file path.
            string modelpath = "";
            if (!DA.GetData(0, ref modelpath))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Model file path not provided.");
                return;
            }
            if (string.IsNullOrEmpty(modelpath) || !File.Exists(modelpath))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Invalid model file path.");
                return;
            }

            // 2. If the model has changed, mark that dynamic inputs need updating.
            if (modelpath != currentModelPath)
            {
                currentModelPath = modelpath;
                dynamicInputsUpdated = false;
            }

            // 3. If not yet updated, open the model, read metadata, and update inputs.
            if (!dynamicInputsUpdated)
            {
                try
                {
                    using (var session = new InferenceSession(modelpath))
                    {
                        UpdateDynamicInputs(session);
                    }
                    dynamicInputsUpdated = true;
                    // Expire the solution so that the component re-solves with the new input parameters.
                    ExpireSolution(true);
                    return;
                }
                catch (Exception ex)
                {
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Error, $"Error loading ONNX model: {ex.Message}");
                    return;
                }
            }

            // 4. Now that the dynamic inputs are set, gather their values.
            // We'll reassemble the inputs grouped by their original key.
            var assembledInputs = new Dictionary<string, float[]>();
            foreach (var kvp in inputMapping)
            {
                string origKey = kvp.Key;
                List<string> dynNames = kvp.Value;
                float[] values = new float[dynNames.Count];
                for (int j = 0; j < dynNames.Count; j++)
                {
                    // Find the parameter index by matching its name.
                    int index = -1;
                    for (int k = 1; k < Params.Input.Count; k++) // start at 1 since index 0 is the model file
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

            // 5. Run the ONNX model with the gathered input values.
            try
            {
                using (var session = new InferenceSession(currentModelPath))
                {
                    var inputs = new List<NamedOnnxValue>();
                    foreach (var kvp in assembledInputs)
                    {
                        string origKey = kvp.Key;
                        float[] vals = kvp.Value;
                        // Create a tensor: if more than one value, use shape [1, n]; else [1].
                        int[] shape = (vals.Length > 1) ? new[] { 1, vals.Length } : new[] { 1 };
                        var tensor = new DenseTensor<float>(vals, shape);
                        // IMPORTANT: The key here must match the original ONNX input name.
                        inputs.Add(NamedOnnxValue.CreateFromTensor(origKey, tensor));
                    }
                    using (var results = session.Run(inputs))
                    {
                        var outputTensor = results.First().AsTensor<float>();
                        float prediction = outputTensor[0];
                        DA.SetData(1, prediction);

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

        // This method reads the ONNX model metadata and updates the input parameters.
        // It creates one or more scalar inputs per model input key based on the tensor shape.
        private void UpdateDynamicInputs(InferenceSession session)
        {
            // Remove all inputs except the first one (the Model parameter).
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
                    param.Description = (featureCount > 1) ? $"Input {i + 1} of {origKey} (Type: {kvp.Value.ElementType})"
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
            // Prevent removal of the first parameter ("Model")
            return side == GH_ParameterSide.Input && index > 0;
        }

        public IGH_Param CreateParameter(GH_ParameterSide side, int index)
        {
            // Not used—dynamic parameters are created in UpdateDynamicInputs.
            return null;
        }

        public bool DestroyParameter(GH_ParameterSide side, int index)
        {
            return side == GH_ParameterSide.Input && index > 0;
        }

        public void VariableParameterMaintenance()
        {
            // No additional maintenance needed.
        }

        #endregion IGH_VariableParameterComponent Implementation

        protected override System.Drawing.Bitmap Icon => null;

        public override Guid ComponentGuid => new Guid("84fd73f3-62e5-4245-948f-fd9f9d554eca");
    }
}