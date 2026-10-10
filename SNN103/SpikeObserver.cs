namespace SNN103;

/// <summary>
/// Research and diagnostic module (Experiment 4.1: Capacity and Zones).
/// Passive observer: monitors emergent network dynamics and measures
/// capacity in bits, sparsity, repeatability, and 1-NN discriminability.
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
    /// Distance between two first-spike patterns. Silence (-1) is mapped
    /// to a distant virtual time step to heavily penalize missing spikes.
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
    /// Leave-one-out 1-NN accuracy: tests whether patterns belonging to the
    /// same stimulus cluster together in the temporal pattern space.
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
        Console.WriteLine("=== EMERGENT ZONES (Spike arrival order by stimulus) ===");
        var allZones = new HashSet<string>();
        foreach (var kv in _zonesByStimulus.OrderBy(kv => kv.Key))
        {
            var groups = kv.Value.GroupBy(z => z).OrderByDescending(g => g.Count()).ToList();
            var top = groups[0];
            double consistency = 100.0 * top.Count() / kv.Value.Count;
            Console.WriteLine($"  Stimulus {kv.Key}: dominant zone [{top.Key}] " +
                              $"— repeatability {consistency:F1}% ({top.Count()}/{kv.Value.Count})");
            Console.WriteLine($"    distribution: {string.Join("  ", groups.Take(6).Select(g => $"[{g.Key}]x{g.Count()}"))}");
            foreach (var z in groups.Select(g => g.Key)) allZones.Add(z);
        }

        double bits = Math.Log2(Math.Max(1, allZones.Count));
        double avgSpikes = _spikeCounts.Count == 0 ? 0 : _spikeCounts.Average();
        double sparsity = 100.0 * (1.0 - avgSpikes / _neuronCount); // % neurons that stayed silent

        Console.WriteLine();
        Console.WriteLine("=== MEASUREMENTS ===");
        Console.WriteLine($"  Distinct zones observed   : {allZones.Count}");
        Console.WriteLine($"  Estimated capacity         : {bits:F2} bits (log2 of distinct zones)");
        Console.WriteLine($"  Combinatorial bound (n!)   : {_neuronCount}! = {Factorial(_neuronCount)} possible orders");
        Console.WriteLine($"  Average spikes / trial     : {avgSpikes:F2} of {_neuronCount} neurons");
        Console.WriteLine($"  Sparsity (silence)         : {sparsity:F1}% silent neurons");
        Console.WriteLine($"  => Each inference used ~{avgSpikes:F1} spikes: actual event budget.");
        Console.WriteLine($"  1-NN Discriminability      : {NearestNeighborAccuracy():F1}%");
    }

    private static long Factorial(int n)
    {
        long r = 1;
        for (int i = 2; i <= n; i++) r *= i;
        return r;
    }
}
