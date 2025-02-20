using Grasshopper;
using Grasshopper.Kernel;
using Rhino.Geometry;
using System;
using System.Collections.Generic;

//using Python.Runtime;
using Microsoft.ML.OnnxRuntime;
using System.IO;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Linq;

namespace BuildingEnergyMLVIP
{
    public class BuildingEnergyMLVIPComponent : GH_Component
    {
        public override GH_Exposure Exposure
        {
            get { return GH_Exposure.primary; }
        }

        /// <summary>
        /// Each implementation of GH_Component must provide a public
        /// constructor without any arguments.
        /// Category represents the Tab in which the component will appear,
        /// Subcategory the panel. If you use non-existing tab or panel names,
        /// new tabs/panels will automatically be created.
        /// </summary>
        public BuildingEnergyMLVIPComponent()
          : base("BuildingEnergyMLVIP", "MLVIP",
            "Description",
            "Params", "General")  // Params -> BuildingEnergyMLVIP
        {
        }

        /// <summary>
        /// Registers all the input parameters for this component.
        /// </summary>
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddTextParameter("Model", "M", "Model", GH_ParamAccess.item);
            pManager.AddTextParameter("Type", "T", "Type of building", GH_ParamAccess.item);
            pManager.AddIntegerParameter("Shape", "S", "Shape of building", GH_ParamAccess.item);
            pManager.AddIntegerParameter("Orientation", "O", "Orientation of the building", GH_ParamAccess.item);
            pManager.AddNumberParameter("Height", "H", "Building height", GH_ParamAccess.item);
            pManager.AddIntegerParameter("Stories", "St", "No. of Storeys", GH_ParamAccess.item);
            pManager.AddNumberParameter("WallArea", "WA", "Wall area of Building", GH_ParamAccess.item);
            pManager.AddNumberParameter("WindowArea", "WinA", "Window area of the Building", GH_ParamAccess.item);
            pManager.AddNumberParameter("RoofArea", "RA", "Roof area of the building", GH_ParamAccess.item);
            pManager.AddTextParameter("EnergyCode", "EC", "Energy code applied", GH_ParamAccess.item);
            pManager.AddTextParameter("HVAC", "HV", "HVAC System", GH_ParamAccess.item);

            pManager[0].Optional = true;
            pManager[1].Optional = true;
            pManager[2].Optional = true;
            pManager[3].Optional = true;
            pManager[4].Optional = true;
            pManager[5].Optional = true;
            pManager[6].Optional = true;
            pManager[7].Optional = true;
            pManager[8].Optional = true;
            pManager[9].Optional = true;
            pManager[10].Optional = true;
        }

        /// <summary>
        /// Registers all the output parameters for this component.
        /// </summary>
        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            //  pManager.AddTextParameter("Input", "I", "Input", GH_ParamAccess.list);
            pManager.AddTextParameter("Output", "O", "Output", GH_ParamAccess.list);
            //pManager.AddNumberParameter("Cooling", "C", "The energy consume for cooling", GH_ParamAccess.item);
            //pManager.AddNumberParameter("Heating", "H", "The energy consume for heating", GH_ParamAccess.item);
        }

        /// <summary>
        /// This is the method that actually does the work.
        /// </summary>
        /// <param name="DA">The DA object can be used to retrieve data from input parameters and
        /// to store data in output parameters.</param>
        protected override void SolveInstance(IGH_DataAccess DA)
        {
            // 1. Read Grasshopper inputs
            string modelpath = "";
            string type = "";
            int shape = 5;
            int orientation = 4;
            double height = 3.0;
            double stories = 1.0;
            double wallarea = 3.0;
            double windowarea = 1.0;
            double roofarea = 2.0;
            string energycode = "";
            string hvac = "";

            DA.GetData(0, ref modelpath);
            DA.GetData(1, ref type);
            DA.GetData(2, ref shape);
            DA.GetData(3, ref orientation);
            DA.GetData(4, ref height);
            DA.GetData(5, ref stories);
            DA.GetData(6, ref wallarea);
            DA.GetData(7, ref windowarea);
            DA.GetData(8, ref roofarea);
            DA.GetData(9, ref energycode);
            DA.GetData(10, ref hvac);

            // 2. Construct the full path to the ONNX model
            string fullPath = Path.Combine(modelpath);
            Console.WriteLine("Full Path to ONNX Model: " + fullPath);

            // 3. Create the InferenceSession (no 'using' statements)
            InferenceSession session = null;
            try
            {
                session = new InferenceSession(fullPath);

                // 4. Prepare input data (example: 7 numeric features)
                float shapeF = shape;
                float orientationF = orientation;
                float heightF = (float)height;
                float storiesF = (float)stories;
                float wallareaF = (float)wallarea;
                float windowareaF = (float)windowarea;
                float roofareaF = (float)roofarea;

                float[] inputData = new float[]
                {
                shapeF,
                orientationF,
                heightF,
                storiesF,
                wallareaF,
                windowareaF,
                roofareaF
                };

                // Shape [1, 7] as an example; adjust as needed
                var inputTensor = new DenseTensor<float>(inputData, new[] { 1, inputData.Length });

                var inputs = new List<NamedOnnxValue>
            {
                // "input" must match the actual input name in your ONNX model
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };

                var results = session.Run(inputs);
                try
                {
                    // 5. Retrieve the model inference results
                    //    Adjust for your actual output layer name(s)
                    foreach (var result in results)
                    {
                        if (result.Name == "output")
                        {
                            var outputTensor = result.AsTensor<float>();
                            float[] outputVals = outputTensor.ToArray();

                            Console.WriteLine("=== Model Output ===");
                            for (int i = 0; i < outputVals.Length; i++)
                            {
                                Console.WriteLine("  " + outputVals[i]);
                            }
                        }
                    }
                }
                finally
                {
                    if (results != null)
                    {
                        results.Dispose();
                    }
                }

                // 6. Get model OUTPUT metadata and set it as GH output
                var outputData = new List<string>();
                foreach (var output in session.OutputMetadata)
                {
                    outputData.Add(
                        string.Format("  {0}: {1} - {2}",
                            output.Key,
                            output.Value.ElementType,
                            string.Join("x", output.Value.Dimensions))
                    );
                }

                // This assumes you've set up an output parameter in your Grasshopper component (index 0)
                DA.SetDataList(0, outputData);
            }
            finally
            {
                // 7. Dispose of the session
                if (session != null)
                {
                    session.Dispose();
                }
            }
        }

        /// <summary>
        /// Provides an Icon for every component that will be visible in the User Interface.
        /// Icons need to be 24x24 pixels.
        /// You can add image files to your project resources and access them like this:
        /// return Resources.IconForThisComponent;
        /// </summary>
        protected override System.Drawing.Bitmap Icon => null;

        /// <summary>
        /// Each component must have a unique Guid to identify it.
        /// It is vital this Guid doesn't change otherwise old ghx files
        /// that use the old ID will partially fail during loading.
        /// </summary>
        public override Guid ComponentGuid => new Guid("84fd73f3-62e5-4245-948f-fd9f9d554eca");
    }
}