using System.Globalization;

namespace SNN103;

/// <summary>
/// Supervised experiment: tests whether the spiking network learns an input->output mapping from CSV.
/// Format: CSV with 6 columns (4 binary inputs + 2 binary targets per row).
///
/// Architecture (ELM-style, deterministic and reproducible):
///   - Hidden layer: 96 LIF neurons with FIXED RANDOM weights.
///     Expands the 4 inputs into a sparse 96-dimensional code.
///     No STDP: fixed random projection separates cleanly.
///   - Readout: 2 supervised outputs trained with the Perceptron rule.
/// </summary>
public static class CsvExperiment
{
    public static void Run(string csvPath)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        Console.WriteLine("SNN103 — Supervised Learning from CSV");
        Console.WriteLine($"File: {csvPath}");
        Console.WriteLine();

        // ---- 1) Load CSV ----
        var rows = new List<(double[] x, int[] y)>();
        foreach (var line in File.ReadAllLines(csvPath))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var p = line.Split(',')
                        .Select(s => double.Parse(s.Trim(), CultureInfo.InvariantCulture))
                        .ToArray();
            if (p.Length != 6)
                throw new Exception($"Expected 6 columns, found {p.Length} in: {line}");
            rows.Add((p[0..4], new[] { (int)p[4], (int)p[5] }));
        }
        Console.WriteLine($"Rows: {rows.Count} (inputs: 4, outputs: 2)");
        Console.WriteLine();

        const int hiddenNeurons = 96;
        var rng = new Random(7);

        // Hidden layer: fixed random expansion.
        var layer = new CompetitiveLayer(neuronCount: hiddenNeurons, inputCount: 4, seed: 7);
        layer.PresenceCoding = true;   // binary presence coding
        layer.TrialSteps = 32;
        layer.InputGain = 4.0;
        layer.InputJitter = 0;         // clean evaluation
        layer.LearningEnabled = false; // frozen weights

        // Precalculate hidden feature responses once
        var hiddenFeatures = rows.Select(r =>
        {
            var tr = layer.PresentStimulus(r.x, noiseAmp: 0.0);
            return FeatureVector(tr, hiddenNeurons);
        }).ToList();

        // ---- Phase 2: Supervised perceptron (frozen hidden layer) ----
        var readout = new ReadoutLayer(inputCount: hiddenNeurons, outputCount: 2, seed: 7);
        Console.WriteLine("Phase 2: Supervised perceptron (frozen hidden layer)...");
        var order = Enumerable.Range(0, rows.Count).ToArray();
        int convergedAt = -1;
        for (int epoch = 0; epoch < 10000; epoch++)
        {
            Shuffle(order, rng);
            double err = 0;
            foreach (int i in order)
            {
                err += readout.Train(hiddenFeatures[i], rows[i].y);
            }
            if (err == 0) { convergedAt = epoch; break; }
        }
        Console.WriteLine(convergedAt >= 0
            ? $"Phase 2 completed: converged at epoch {convergedAt}."
            : "Phase 2: Did not converge within 10000 epochs.");
        Console.WriteLine();

        // ---- Phase 3: Evaluation ----
        Console.WriteLine("Phase 3: Clean evaluation");
        Console.WriteLine("  in1 in2 in3 in4 | exp1 exp2 | got1 got2 |");
        int ok = 0;
        foreach (var (x, y) in rows)
        {
            var r = layer.PresentStimulus(x, noiseAmp: 0.0);
            int[] pred = readout.Predict(FeatureVector(r, hiddenNeurons));
            bool good = pred[0] == y[0] && pred[1] == y[1];
            if (good) ok++;
            Console.WriteLine($"   {x[0]}   {x[1]}   {x[2]}   {x[3]}  |   {y[0]}    {y[1]}   |   {pred[0]}    {pred[1]}   |  {(good ? "OK" : "FAIL")}");
        }
        Console.WriteLine();
        Console.WriteLine($"Accuracy: {ok}/{rows.Count} = {100.0 * ok / rows.Count:F1}%");
        Console.WriteLine(ok == rows.Count
            ? "Network successfully learned the mapping."
            : "Mapping incomplete: consider increasing hidden neurons.");
    }

    /// <summary>Binary feature vector: 1.0 if neuron spiked during presentation.</summary>
    private static double[] FeatureVector(TrialResult r, int n)
    {
        var v = new double[n];
        foreach (int i in r.SpikeOrder) v[i] = 1.0;
        return v;
    }

    private static void Shuffle(int[] a, Random rng)
    {
        for (int i = a.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (a[i], a[j]) = (a[j], a[i]);
        }
    }
}
