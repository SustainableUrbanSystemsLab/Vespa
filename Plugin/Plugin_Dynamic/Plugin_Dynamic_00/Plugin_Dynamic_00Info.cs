using System;
using System.Drawing;
using Grasshopper;
using Grasshopper.Kernel;

namespace Plugin_Dynamic_00
{
  public class Plugin_Dynamic_00Info : GH_AssemblyInfo
  {
    public override string Name => "Plugin_Dynamic_00";

    //Return a 24x24 pixel bitmap to represent this GHA library.
    public override Bitmap Icon => null;

    //Return a short string describing the purpose of this GHA library.
    public override string Description => "";

    public override Guid Id => new Guid("69fe7baf-c30b-4b82-b21e-db3cc256751b");

    //Return a string identifying you or your company.
    public override string AuthorName => "";

    //Return a string representing your preferred contact details.
    public override string AuthorContact => "";

    //Return a string representing the version.  This returns the same version as the assembly.
    public override string AssemblyVersion => GetType().Assembly.GetName().Version.ToString();
  }
}