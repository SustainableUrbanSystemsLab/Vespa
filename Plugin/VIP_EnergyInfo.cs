using System;
using System.Drawing;
using Grasshopper;
using Grasshopper.Kernel;

namespace VIP_Energy
{
  public class VIP_EnergyInfo : GH_AssemblyInfo
  {
    public override string Name => "VIP_Energy";

    //Return a 24x24 pixel bitmap to represent this GHA library.
    public override Bitmap Icon => null;

    //Return a short string describing the purpose of this GHA library.
    public override string Description => "";

    public override Guid Id => new Guid("e5ed897a-7c14-4124-b636-4f4b34881f7a");

    //Return a string identifying you or your company.
    public override string AuthorName => "";

    //Return a string representing your preferred contact details.
    public override string AuthorContact => "";

    //Return a string representing the version.  This returns the same version as the assembly.
    public override string AssemblyVersion => GetType().Assembly.GetName().Version.ToString();
  }
}