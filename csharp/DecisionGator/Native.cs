// P/Invoke declarations matching code/include/decisiongator.h. Applications use Decisions instead.
using System.Runtime.InteropServices;

namespace DecisionGator;

internal static unsafe class Native
{
    internal const string Library = "decisiongator";

    [StructLayout(LayoutKind.Sequential)]
    internal struct Criteria
    {
        public byte* Yes; public nuint YesBytes;
        public byte* No; public nuint NoBytes;
    }

    [DllImport(Library, EntryPoint = "dg_is_yes_p")]
    internal static extern int IsYesP(byte* content, nuint contentBytes, byte* question, nuint questionBytes, Criteria* criteria, out double pYes);

    [DllImport(Library, EntryPoint = "dg_is_yes_at_threshold")]
    internal static extern int IsYesAtThreshold(byte* content, nuint contentBytes, byte* question, nuint questionBytes, Criteria* criteria, double threshold, out byte yes);

    [DllImport(Library, EntryPoint = "dg_choose_p")]
    internal static extern int ChooseP(byte* content, nuint contentBytes, byte* question, nuint questionBytes, byte** options, nuint* optionBytes, nuint optionCount, Criteria* criteria, int* outIndex, double* outP);

    [DllImport(Library, EntryPoint = "dg_last_error")]
    internal static extern int LastError(byte* buffer, nuint capacity, out nuint required);

    static Native()
    {
        NativeLibrary.SetDllImportResolver(typeof(Native).Assembly, Resolve);
    }

    // Load the library by absolute path from the bundle directory so it can find
    // its model files beside itself (the header asks for this on Unix).
    private static IntPtr Resolve(string name, System.Reflection.Assembly assembly, DllImportSearchPath? path)
    {
        if (name != Library) return IntPtr.Zero;
        string file = OperatingSystem.IsWindows() ? "decisiongator.dll"
                    : OperatingSystem.IsMacOS() ? "libdecisiongator.dylib"
                    : "libdecisiongator.so";
        // The one-file build (code/standalone) has the model inside and a different name.
        string standalone = OperatingSystem.IsWindows() ? "decisiongator_standalone.dll"
                          : OperatingSystem.IsMacOS() ? "libdecisiongator_standalone.dylib"
                          : "libdecisiongator_standalone.so";
        foreach (string directory in Bundle.Candidates())
        {
            foreach (string candidate in new[] { standalone, file })
            {
                string full = Path.Combine(directory, candidate);
                if (File.Exists(full)) return NativeLibrary.Load(full);
            }
        }
        throw new DllNotFoundException(
            $"{file} not found. Set DECISIONGATOR_BUNDLE or Bundle.Directory to the bundle folder, " +
            "or copy the bundle into a 'decisiongator' folder next to the application.");
    }

    internal static string LastErrorMessage()
    {
        LastError(null, 0, out nuint required);
        if (required == 0) return "unknown error";
        var buffer = new byte[(int)required];
        fixed (byte* b = buffer) LastError(b, required, out required);
        return System.Text.Encoding.UTF8.GetString(buffer, 0, Math.Max(0, (int)required - 1));
    }
}
