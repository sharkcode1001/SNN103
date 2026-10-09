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
        myPlot.Title("SNN103 vs MLP Clásico: Precisión y Cómputo (4 clases)");

        double[] accuracies = { snnAcc, mlpAcc };
        string[] labels = { $"SNN103\n(~{snnSpikes:F1} Spikes/inf)", $"MLP Clásico\n(~{mlpMults:F0} MACs/inf)" };

        var barPlot = myPlot.Add.Bars(accuracies);
        barPlot.LegendText = "Precisión Test (%)";
        
        myPlot.Axes.Bottom.TickGenerator = new ScottPlot.TickGenerators.NumericManual(
            new[] { 0.0, 1.0 },
            labels
        );

        myPlot.Axes.Left.Label.Text = "Precisión (%)";
        myPlot.Axes.SetLimitsY(0, 100);

        string filePath = Path.Combine(outputFolder, "snn_vs_mlp_comparison.png");
        myPlot.SavePng(filePath, 800, 600);
        Console.WriteLine($"[Gráfica Exportada] {filePath}");
    }

    public static void ExportConfusionMatrix(int[,] matrix, int numClasses, string outputFolder = "plots")
    {
        Directory.CreateDirectory(outputFolder);

        var myPlot = new ScottPlot.Plot();
        myPlot.Title("Matriz de Confusión - SNN103 (Datos de Prueba)");

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

        myPlot.Axes.Left.Label.Text = "Clase Real";
        myPlot.Axes.Bottom.Label.Text = "Clase Predicha";

        string filePath = Path.Combine(outputFolder, "confusion_matrix.png");
        myPlot.SavePng(filePath, 700, 600);
        Console.WriteLine($"[Gráfica Exportada] {filePath}");
    }

    public static void ExportRasterPlot(double[] spikeTimes, double[] neuronIndices, string outputFolder = "plots")
    {
        Directory.CreateDirectory(outputFolder);

        var myPlot = new ScottPlot.Plot();
        myPlot.Title("Raster Plot - Trama Temporal de Impulsos (Spikes)");

        var scatter = myPlot.Add.ScatterPoints(spikeTimes, neuronIndices);
        scatter.MarkerSize = 6;
        scatter.Color = ScottPlot.Colors.Navy;

        myPlot.Axes.Bottom.Label.Text = "Pasos de Simulación / Tiempo (t)";
        myPlot.Axes.Left.Label.Text = "Índice de Neurona / Grupo";

        string filePath = Path.Combine(outputFolder, "raster_plot.png");
        myPlot.SavePng(filePath, 800, 500);
        Console.WriteLine($"[Gráfica Exportada] {filePath}");
    }

    public static void ExportBenchmark10Classes(
        List<double> accOrder, 
        List<double> accBin, 
        List<double> accMlp, 
        string outputFolder = "plots")
    {
        Directory.CreateDirectory(outputFolder);

        var myPlot = new ScottPlot.Plot();
        myPlot.Title("Ablación Central: Orden vs Binario vs MLP (10 Clases - 5 Semillas)");

        double meanOrder = accOrder.Average();
        double meanBin = accBin.Average();
        double meanMlp = accMlp.Average();

        double[] values = { meanOrder, meanBin, meanMlp };
        string[] labels = { 
            $"SNN Orden\n{meanOrder:F1}% ± {Std(accOrder):F1}", 
            $"SNN Binario\n{meanBin:F1}% ± {Std(accBin):F1}", 
            $"MLP Clásico\n{meanMlp:F1}% ± {Std(accMlp):F1}" 
        };

        var bars = myPlot.Add.Bars(values);
        bars.LegendText = "Precisión Media Test (%)";

        myPlot.Axes.Bottom.TickGenerator = new ScottPlot.TickGenerators.NumericManual(
            new[] { 0.0, 1.0, 2.0 },
            labels
        );

        myPlot.Axes.Left.Label.Text = "Precisión en Test Set (%)";
        myPlot.Axes.SetLimitsY(0, 105);

        string filePath = Path.Combine(outputFolder, "benchmark_10classes_ablation.png");
        myPlot.SavePng(filePath, 850, 600);
        Console.WriteLine($"[Gráfica Exportada] {filePath}");
    }

    private static double Std(List<double> v)
    {
        double m = v.Average();
        return Math.Sqrt(v.Sum(x => (x - m) * (x - m)) / (v.Count - 1));
    }
}
