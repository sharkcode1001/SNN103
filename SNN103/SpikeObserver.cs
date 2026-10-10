namespace SNN103;

/// <summary>
/// Módulo de investigación y diagnóstico (Experimentos 4.1 de Capacidad y Zonas).
/// Observador pasivo: NO codifica ni decodifica nada, solo observa las dinámicas
/// emergentes de la red y mide capacidad en bits, dispersión, repetibilidad y discriminabilidad 1-NN.
/// </summary>
public sealed class SpikeObserver
{
    private readonly int _neuronCount;
    private readonly int _trialSteps;
    private readonly Dictionary<int, List<string>> _zonesByStimulus = new();
    private readonly List<(int stimulus, int[] times)> _patterns = new();
    private readonly List<int> _spikeCounts = new();

    public SpikeObserver(int neuronCount, int trialSteps)
    {
        _neuronCount = neuronCount;
        _trialSteps = trialSteps;
    }

    public void Record(int stimulusId, TrialResult r)
    {
        if (!_zonesByStimulus.TryGetValue(stimulusId, out var list))
            _zonesByStimulus[stimulusId] = list = new List<string>();
        list.Add(r.Zone);
        _patterns.Add((stimulusId, (int[])r.FirstSpikeTimes.Clone()));
        _spikeCounts.Add(r.SpikeOrder.Count);
    }

    /// <summary>
    /// Distancia entre dos patrones de primer disparo. Silencio (-1) se mapea
    /// a un tiempo ficticio lejano: disparar-vs-no-disparar penaliza fuerte.
    /// </summary>
    private double PatternDistance(int[] a, int[] b)
    {
        double d = 0;
        for (int i = 0; i < _neuronCount; i++)
        {
            int ta = a[i] < 0 ? _trialSteps + 10 : a[i];
            int tb = b[i] < 0 ? _trialSteps + 10 : b[i];
            d += Math.Abs(ta - tb);
        }
        return d;
    }

    /// <summary>
    /// Precisión 1-NN leave-one-out: para cada ensayo, ¿el ensayo más parecido
    /// es del mismo estímulo? Es la medida funcional del código: no exige que
    /// el string de la zona sea idéntico, solo que los patrones del mismo
    /// estímulo vivan juntos en el espacio de patrones.
    /// </summary>
    public double NearestNeighborAccuracy()
    {
        if (_patterns.Count < 2) return 0;
        int correct = 0;
        for (int i = 0; i < _patterns.Count; i++)
        {
            double best = double.MaxValue;
            int bestStim = -1;
            for (int j = 0; j < _patterns.Count; j++)
            {
                if (i == j) continue;
                double d = PatternDistance(_patterns[i].times, _patterns[j].times);
                if (d < best) { best = d; bestStim = _patterns[j].stimulus; }
            }
            if (bestStim == _patterns[i].stimulus) correct++;
        }
        return 100.0 * correct / _patterns.Count;
    }

    public void PrintReport()
    {
        Console.WriteLine();
        Console.WriteLine("=== ZONAS EMERGENTES (orden de disparo por estímulo) ===");
        var allZones = new HashSet<string>();
        foreach (var kv in _zonesByStimulus.OrderBy(kv => kv.Key))
        {
            var groups = kv.Value.GroupBy(z => z).OrderByDescending(g => g.Count()).ToList();
            var top = groups[0];
            double consistency = 100.0 * top.Count() / kv.Value.Count;
            Console.WriteLine($"  Estímulo {kv.Key}: zona mayoritaria [{top.Key}] " +
                              $"— repetibilidad {consistency:F1}% ({top.Count()}/{kv.Value.Count})");
            Console.WriteLine($"    distribución: {string.Join("  ", groups.Take(6).Select(g => $"[{g.Key}]x{g.Count()}"))}");
            foreach (var z in groups.Select(g => g.Key)) allZones.Add(z);
        }

        double bits = Math.Log2(Math.Max(1, allZones.Count));
        double avgSpikes = _spikeCounts.Count == 0 ? 0 : _spikeCounts.Average();
        double sparsity = 100.0 * (1.0 - avgSpikes / _neuronCount); // % que permaneció en silencio

        Console.WriteLine();
        Console.WriteLine("=== MEDICIONES ===");
        Console.WriteLine($"  Zonas distintas observadas : {allZones.Count}");
        Console.WriteLine($"  Capacidad estimada         : {bits:F2} bits (log2 de zonas distintas)");
        Console.WriteLine($"  Cota combinatoria (n!)     : {_neuronCount}! = {Factorial(_neuronCount)} órdenes posibles");
        Console.WriteLine($"  Spikes promedio / ensayo  : {avgSpikes:F2} de {_neuronCount} neuronas");
        Console.WriteLine($"  Dispersión (silencio)      : {sparsity:F1}% de las neuronas no dispararon");
        Console.WriteLine($"  => Cada inferencia usó ~{avgSpikes:F1} spikes: ese es el cómputo real.");
        Console.WriteLine($"  Discriminabilidad 1-NN    : {NearestNeighborAccuracy():F1}% " +
                          $"(¿el patrón identifica al estímulo?)");
    }

    private static long Factorial(int n)
    {
        long r = 1;
        for (int i = 2; i <= n; i++) r *= i;
        return r;
    }
}
