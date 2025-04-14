using Grasshopper;
using Grasshopper.Kernel;
using Rhino.Geometry;
using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using System.IO;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Linq;
using Python.Runtime;

namespace BuildingEnergyMLVIP
{
    // This component implements IGH_VariableParameterComponent so that its input parameters can be changed dynamically.
    public class BuildingEnergyMLVIPComponent : GH_Component, IGH_VariableParameterComponent
    {
        // Store the current model file path.
        private string currentModelPath = "";

        private bool dynamicInputsUpdated = false;

        // This list holds the names of all dynamically created parameters.
        private List<string> modelInputNames = new List<string>();

        // Maps each ONNX input key (for example, "input") to a list of dynamic parameter names.
        private Dictionary<string, List<string>> inputMapping = new Dictionary<string, List<string>>();

        private static readonly string pythonDir = @"C:\Users\josep\AppData\Local\Programs\Python\Python312";

        // Static constructor: runs once before the first instance is created.
        static BuildingEnergyMLVIPComponent()
        {
            // (a) Update PATH so python312.dll can be found:
            string existingPath = Environment.GetEnvironmentVariable("PATH") ?? "";
            Environment.SetEnvironmentVariable("PATH", pythonDir + ";" + existingPath);

            // (b) Tell pythonnet which Python to load:
            Runtime.PythonDLL = Path.Combine(pythonDir, "python312.dll");
            PythonEngine.PythonHome = pythonDir;
            PythonEngine.PythonPath = Path.Combine(pythonDir, "Lib", "site-packages");

            // (c) Initialize pythonnet:
            PythonEngine.Initialize();
        }

        public BuildingEnergyMLVIPComponent()
          : base("BuildingEnergyMLVIP", "MLVIP",
                "Predict building energy load using a joblib model with dynamic inputs",
                "Params", "General")
        {
        }

        public override GH_Exposure Exposure => GH_Exposure.primary;

        // Initially, only the Model file path is registered.
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddTextParameter("Model", "M", "Path to joblib model file", GH_ParamAccess.item);
            // Dynamic inputs will be added later based on the model metadata.
        }

        // Output 0: Model output metadata; Output 1: Model prediction.
        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddTextParameter("Model Metadata", "MM", "Feature names from the model", GH_ParamAccess.list);
            pManager.AddNumberParameter("Prediction", "P", "Predicted energy load", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            Environment.SetEnvironmentVariable("PYTHONNET_PYDLL", @"C:\Users\josep\AppData\Local\Programs\Python\Python312\python312.dll");
            Environment.SetEnvironmentVariable("PYTHONHOME", @"C:\Users\josep\AppData\Local\Programs\Python\Python312");
            Environment.SetEnvironmentVariable("PYTHONPATH", @"C:\Users\josep\AppData\Local\Programs\Python\Python312\Lib\site-packages");

            // Always do this before calling PythonEngine.Initialize() or using (Py.GIL()).
            string existingPath = Environment.GetEnvironmentVariable("PATH") ?? "";
            Environment.SetEnvironmentVariable("PATH", pythonDir + ";" + existingPath);

            // 1. Get the model file path.
            string modelPath = "";
            if (!DA.GetData(0, ref modelPath))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Model file path not provided.");
                return;
            }
            if (string.IsNullOrEmpty(modelPath) || !File.Exists(modelPath))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Invalid model file path.");
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
                using (Py.GIL())
                {
                    dynamic joblib = Py.Import("joblib");
                    dynamic model = null;
                    try
                    {
                        model = joblib.load(modelPath);
                    }
                    catch (Exception)
                    {
                        // If joblib.load fails, try cloudpickle.
                        dynamic cloudpickle = Py.Import("cloudpickle");
                        using (var f = File.OpenRead(modelPath))
                        {
                            byte[] bytes = new byte[f.Length];
                            f.Read(bytes, 0, bytes.Length);

                            // Import the builtins module and create an empty Python list using builtins.list()
                            dynamic builtins = Py.Import("builtins");
                            dynamic pyList = builtins.list();

                            // Append each byte wrapped as a Python int to the Python list.
                            foreach (byte b in bytes)
                            {
                                pyList.append(new PyInt(b));
                            }

                            // Call builtins.bytes() on the Python list to create a new python bytes object.
                            PyObject pyBytes = builtins.bytes(pyList);

                            // Load the model using cloudpickle.loads with the Python bytes object.
                            model = cloudpickle.loads(pyBytes);
                        }
                    }
                    List<string> featureNames = new List<string>();
                    if (model.HasAttr("feature_names_in_"))
                    {
                        dynamic features = model.feature_names_in_;
                        foreach (PyObject item in features)
                        {
                            featureNames.Add(item.ToString());
                        }
                    }
                    else
                    {
                        AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Model does not have a feature_names_in_ attribute.");
                        return;
                    }

                    // Remove any dynamic inputs (keeping the first "Model" input).
                    while (Params.Input.Count > 1)
                        Params.UnregisterInputParameter(Params.Input[1], true);
                    modelInputNames.Clear();
                    inputMapping.Clear();

                    // For each feature, create a dynamic scalar input.
                    foreach (var feature in featureNames)
                    {
                        string paramName = feature;
                        var param = new Grasshopper.Kernel.Parameters.Param_Number();
                        param.Name = paramName;
                        param.NickName = paramName;
                        param.Access = GH_ParamAccess.item;
                        param.Optional = true;
                        param.Description = $"Input for feature {feature}";
                        Params.RegisterInputParam(param);
                        modelInputNames.Add(paramName);
                        inputMapping[feature] = new List<string> { paramName };
                    }
                }
                dynamicInputsUpdated = true;
                ExpireSolution(true);
                return;
            }

            // 4. Gather dynamic input values.
            var assembledInputs = new Dictionary<string, float[]>();
            foreach (var kvp in inputMapping)
            {
                string feature = kvp.Key;
                List<string> dynNames = kvp.Value;
                float[] values = new float[dynNames.Count];
                for (int j = 0; j < dynNames.Count; j++)
                {
                    // Find the parameter index by matching the dynamic parameter's name.
                    int index = -1;
                    for (int k = 1; k < Params.Input.Count; k++) // skip index 0 as it is the Model input.
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
                assembledInputs[feature] = values;
            }

            // 5. Build the input vector in the order defined by feature_names_in_.
            List<float> inputVector = new List<float>();
            using (Py.GIL())
            {
                dynamic joblib = Py.Import("joblib");
                dynamic model = joblib.load(modelPath);
                dynamic features = model.feature_names_in_;
                foreach (PyObject item in features)
                {
                    string feature = item.ToString();
                    if (assembledInputs.ContainsKey(feature))
                    {
                        // We assume one scalar per feature.
                        inputVector.Add(assembledInputs[feature][0]);
                    }
                    else
                    {
                        AddRuntimeMessage(GH_RuntimeMessageLevel.Error, $"Missing input for feature {feature}");
                        return;
                    }
                }
            }

            // 6. Call model.predict using Python.
            float prediction = 0;
            using (Py.GIL())
            {
                dynamic np = Py.Import("numpy");
                // Create a 2D numpy array of shape [1, n]
                PyObject npInput = np.array(inputVector.ToArray()).reshape(1, inputVector.Count);
                dynamic joblib = Py.Import("joblib");
                dynamic model = joblib.load(modelPath);
                dynamic result = model.predict(npInput);
                prediction = (float)result[0];
            }

            DA.SetData(1, prediction);
            // Output the model's feature names as metadata.
            DA.SetDataList(0, inputMapping.Keys);
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