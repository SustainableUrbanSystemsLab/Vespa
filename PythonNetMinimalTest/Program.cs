using System;
using Python.Runtime;

namespace PythonNetMinimalTest
{
    internal class Program
    {
        // Static constructor to ensure environment variables are set before Python.Runtime is initialized.
        static Program()
        {
            // Update these paths to match your Python 3.12 installation.
            Environment.SetEnvironmentVariable("PYTHONNET_PYDLL", @"C:\Users\josep\AppData\Local\Programs\Python\Python312\python312.dll");
            Environment.SetEnvironmentVariable("PYTHONHOME", @"C:\Users\josep\AppData\Local\Programs\Python\Python312");
            Environment.SetEnvironmentVariable("PYTHONPATH", @"C:\Users\josep\AppData\Local\Programs\Python\Python312\Lib\site-packages");
            Console.WriteLine("Static constructor: Environment variables set for Python 3.12.");
        }

        private static void Main(string[] args)
        {
            try
            {
                Console.WriteLine("Acquiring GIL...");
                using (Py.GIL())
                {
                    Console.WriteLine("GIL acquired.");
                    dynamic np = Py.Import("numpy");
                    Console.WriteLine("numpy version: " + np.__version__);
                    // Verify that numpy.core is accessible.
                    dynamic npCore = np.core;
                    Console.WriteLine("Successfully accessed numpy.core.");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error during Python interop: " + ex);
            }

            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }
    }
}