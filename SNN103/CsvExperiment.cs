using System.Globalization;

namespace SNN103;

/// <summary>
/// Experimento supervisado v2: ¿puede la red aprender un mapeo entradas->salidas?
/// Usa un CSV de 6 columnas (4 entradas + 2 salidas por fila).
///
/// Arquitectura (estilo ELM, honesta y reproducible):
///   - Capa oculta: 96 neuronas de pulsos con pesos ALEATORIOS FIJOS.
///     Expande las 4 entradas binarias a un código disperso de 96 dimensiones.
///     Sin STDP aquí: la proyección aleatoria ya separa bien.
///   - Readout: 2 salidas entrenadas con REGLA DEL PERCEPTRÓN (supervisado).
///     Por el teorema de convergencia: si los códigos son separables, converge.
///
/// La capa oculta usa codificación por presencia (un canal binario en 0 no
/// emite spike): el silencio también es información.
///
/// Lecciones del desarrollo (2026-10-04):
///   - El STDP debe estar APAGADO durante lo supervisado: si está activo,
///     mueve los pesos después de que el perceptrón converge y lo sabotea.
///   - La "co-adaptación" (STDP + perceptrón simultáneos) dio 100% una vez
///     por accidente, pero es inestable: depende de la trayectoria.
///   - Pesos congelados + perceptrón: 100% determinista y reproducible.
/// </summary>
public static class CsvExperiment
{
    public static void Run(string csvPath)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        Console.WriteLine("SNN103 v2 — aprendizaje supervisado desde CSV");
        Console.WriteLine($"Fichero: {csvPath}");
        Console.WriteLine();

        // ---- 1) Cargar CSV ----
        var rows = new List<(double[] x, int[] y)>();
        foreach (var line in File.ReadAllLines(csvPath))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var p = line.Split(',')
                        .Select(s => double.Parse(s.Trim(), CultureInfo.InvariantCulture))
                        .ToArray();
            if (p.Length != 6)
                throw new Exception($"Se esperaban 6 columnas y hay {p.Length} en: {line}");
            rows.Add((p[0..4], new[] { (int)p[4], (int)p[5] }));
        }
        Console.WriteLine($"Filas: {rows.Count} (entradas: 4, salidas: 2)");
        Console.WriteLine();

        const int hiddenNeurons = 96;
        var rng = new Random(7);

        // Capa oculta: expansión aleatoria fija. STDP apagado desde el inicio.
        var layer = new CompetitiveLayer(neuronCount: hiddenNeurons, inputCount: 4, seed: 7);
        layer.PresenceCoding = true;   // binario: el 0 no emite spike
        layer.TrialSteps = 32;
        layer.InputGain = 4.0;         // un solo spike fuerte debe bastar para disparar (commit=3)
        layer.InputJitter = 0;         // canal limpio: el experimento es de aprendizaje, no de ruido
        layer.LearningEnabled = false; // pesos fijos: la representación no se mueve

        // ---- Fase 2: perceptrón supervisado (capa oculta congelada) ----
        var readout = new ReadoutLayer(inputCount: hiddenNeurons, outputCount: 2, seed: 7);
        Console.WriteLine("Fase 2: perceptrón supervisado (capa oculta congelada)...");
        var order = Enumerable.Range(0, rows.Count).ToArray();
        int convergedAt = -1;
        for (int epoch = 0; epoch < 10000; epoch++)
        {
            Shuffle(order, rng);
            double err = 0;
            foreach (int i in order)
            {
                var r = layer.PresentStimulus(rows[i].x, noiseAmp: 0.0);
                err += readout.Train(FeatureVector(r, hiddenNeurons), rows[i].y);
            }
            if (err == 0) { convergedAt = epoch; break; }
        }
        Console.WriteLine(convergedAt >= 0
            ? $"Fase 2 completa: convergió en la época {convergedAt}."
            : "Fase 2: NO convergió en 10000 épocas.");
        Console.WriteLine();

        // ---- Fase 3: evaluación ----
        Console.WriteLine("Fase 3: evaluación (presentación limpia)");
        Console.WriteLine("  in1 in2 in3 in4 | esp1 esp2 | obt1 obt2 |");
        int ok = 0;
        foreach (var (x, y) in rows)
        {
            var r = layer.PresentStimulus(x, noiseAmp: 0.0);
            int[] pred = readout.Predict(FeatureVector(r, hiddenNeurons));
            bool good = pred[0] == y[0] && pred[1] == y[1];
            if (good) ok++;
            Console.WriteLine($"   {x[0]}   {x[1]}   {x[2]}   {x[3]}  |   {y[0]}    {y[1]}   |   {pred[0]}    {pred[1]}   |  {(good ? "ok" : "FALLO")}");
        }
        Console.WriteLine();
        Console.WriteLine($"Precisión: {ok}/{rows.Count} = {100.0 * ok / rows.Count:F1}%");
        Console.WriteLine(ok == rows.Count
            ? "La red aprendió el mapeo completo."
            : "No llegó al 100%: probar con más neuronas o revisar los códigos.");
    }

    /// <summary>Vector binario: 1.0 si la neurona disparó en el ensayo.</summary>
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
