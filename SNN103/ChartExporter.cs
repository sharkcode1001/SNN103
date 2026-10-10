using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ScottPlot;

namespace SNN103;

public static class ChartExporter
{
    public static void ExportSnnVsMlpComparison(
        double snnAcc, double mlpAcc, 
        double snnSpikes, double mlpMults, 
        string outputFolder = "plots")
    {
        Directory.CreateDirectory(outputFolder);

        var myPlot = new ScottPlot.Plot();
        myPlot.Title("SNN103 vs Classic MLP: Accuracy & Computation (4 Classes)");

        double[] accuracies = { snnAcc, mlpAcc };
        string[] labels = { $"SNN103\n(~{snnSpikes:F1} Spikes/inf)", $"Classic MLP\n(~{mlpMults:F0} MACs/inf)" };

        var barPlot = myPlot.Add.Bars(accuracies);
        barPlot.LegendText = "Test Accuracy (%)";
        
        myPlot.Axes.Bottom.TickGenerator = new ScottPlot.TickGenerators.NumericManual(
            new[] { 0.0, 1.0 },
            labels
        );

        myPlot.Axes.Left.Label.Text = "Accuracy (%)";
        myPlot.Axes.SetLimitsY(0, 100);

        string filePath = Path.Combine(outputFolder, "snn_vs_mlp_comparison.png");
        myPlot.SavePng(filePath, 800, 600);
        Console.WriteLine($"[Chart Exported] {filePath}");
    }

    public static void ExportConfusionMatrix(int[,] matrix, int numClasses, string outputFolder = "plots")
    {
        Directory.CreateDirectory(outputFolder);

        var myPlot = new ScottPlot.Plot();
        myPlot.Title("Confusion Matrix - SNN103 (Test Set)");

        double[,] values = new double[numClasses, numClasses];
        for (int r = 0; r < numClasses; r++)
        {
            for (int c = 0; c < numClasses; c++)
            {
                values[r, c] = matrix[r, c];
            }
        }

        var heatmap = myPlot.Add.Heatmap(values);
        heatmap.Colormap = new ScottPlot.Colormaps.Viridis();

        myPlot.Axes.Left.Label.Text = "True Class";
        myPlot.Axes.Bottom.Label.Text = "Predicted Class";

        string filePath = Path.Combine(outputFolder, "confusion_matrix.png");
        myPlot.SavePng(filePath, 700, 600);
        Console.WriteLine($"[Chart Exported] {filePath}");
    }

    public static void ExportRasterPlot(double[] spikeTimes, double[] neuronIndices, string outputFolder = "plots")
    {
        Directory.CreateDirectory(outputFolder);

        var myPlot = new ScottPlot.Plot();
        myPlot.Title("Spike Raster Plot - Burst-Driven Temporal Timing");

        var scatter = myPlot.Add.ScatterPoints(spikeTimes, neuronIndices);
        scatter.MarkerSize = 6;
        scatter.Color = ScottPlot.Colors.Navy;

        myPlot.Axes.Bottom.Label.Text = "Simulation Steps / Time (t)";
        myPlot.Axes.Left.Label.Text = "Neuron / Subregion Index";

        string filePath = Path.Combine(outputFolder, "raster_plot.png");
        myPlot.SavePng(filePath, 800, 500);
        Console.WriteLine($"[Chart Exported] {filePath}");
    }

    public static void ExportBenchmark10Classes(
        List<double> accOrder, 
        List<double> accBin, 
        List<double> accMlp, 
        string outputFolder = "plots")
    {
        Directory.CreateDirectory(outputFolder);

        var myPlot = new ScottPlot.Plot();
        myPlot.Title("Core Ablation: Order vs Binary vs MLP (10 Classes - 5 Seeds)");

        double meanOrder = accOrder.Average();
        double meanBin = accBin.Average();
        double meanMlp = accMlp.Average();

        double[] values = { meanOrder, meanBin, meanMlp };
        string[] labels = { 
            $"SNN Order\n{meanOrder:F1}% ± {Std(accOrder):F1}", 
            $"SNN Binary\n{meanBin:F1}% ± {Std(accBin):F1}", 
            $"Classic MLP\n{meanMlp:F1}% ± {Std(accMlp):F1}" 
        };

        var bars = myPlot.Add.Bars(values);
        bars.LegendText = "Mean Test Accuracy (%)";

        myPlot.Axes.Bottom.TickGenerator = new ScottPlot.TickGenerators.NumericManual(
            new[] { 0.0, 1.0, 2.0 },
            labels
        );

        myPlot.Axes.Left.Label.Text = "Test Accuracy (%)";
        myPlot.Axes.SetLimitsY(0, 105);

        string filePath = Path.Combine(outputFolder, "benchmark_10classes_ablation.png");
        myPlot.SavePng(filePath, 850, 600);
        Console.WriteLine($"[Chart Exported] {filePath}");
    }

    private static double Std(List<double> v)
    {
        double m = v.Average();
        return Math.Sqrt(v.Sum(x => (x - m) * (x - m)) / (v.Count - 1));
    }
}
