// DecisionGator from C#. Run from examples/csharp: dotnet run
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DecisionGator;

// Use this checkout's bundle. A deployed application ships the bundle in a
// "decisiongator" folder beside the executable and sets nothing.
Bundle.Directory = Path.Combine(SourceDirectory(), "..", "..", "released",
    RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "windows-x64"
    : RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ? "macos-arm64" : "linux-x64");

var ticket = "Our whole warehouse can't print shipping labels and trucks leave in an hour.";
var question = "Is the customer describing an urgent problem?";

// A yes/no decision at the usual 0.5 cutoff.
Console.WriteLine($"urgent: {Decisions.IsYes(ticket, question).ToString().ToLower()}");

// The probability, so you can keep an uncertain range for a person.
Console.WriteLine($"p_yes = {Decisions.IsYesP(ticket, question):F3}");

// Criteria and a stricter threshold.
var criteria = new Criteria("Work is blocked and there is a deadline within hours.",
                            "The problem is an inconvenience with no near deadline.");
Console.WriteLine($"confident urgent: {Decisions.IsYes(ticket, question, criteria, threshold: 0.90).ToString().ToLower()}");

// Several options instead of yes or no.
string[] teams = { "billing", "technical support", "sales" };
foreach (var choice in Decisions.ChooseP("My card was charged twice for last month's invoice.",
                                         "Which team should handle this message?", teams))
    Console.WriteLine($"{teams[choice.Index],-18} {choice.P:F3}");

static string SourceDirectory([CallerFilePath] string path = "") => Path.GetDirectoryName(path)!;
