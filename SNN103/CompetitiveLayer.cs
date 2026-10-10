namespace SNN103;

/// <summary>
/// Outcome of presenting an input stimulus: first-spike order,
/// spike arrival times, and the resulting quantized temporal "Zone".
/// </summary>
public sealed record TrialResult(List<int> SpikeOrder, int[] FirstSpikeTimes, string Zone);

/// <summary>
/// Competitive spiking neural layer. Neurons receive input spikes,
/// compete through lateral inhibition, and the ORDER in which they fire
/// serves as the emergent internal representation (the B103 line code analogy).
///
/// Input encoding: latency (time-to-first-spike) or presence (binary).
/// When PresenceCoding = true, silent input channels emit no spikes.
/// Plasticity: Pairwise STDP with rank discount (optional / research).
/// </summary>
public sealed class CompetitiveLayer
{
    public int NeuronCount { get; }
    public int InputCount { get; }

    /// <summary>Synaptic weights [neuron, channel]. Initialized uniformly.</summary>
    public double[,] Weights { get; }

    private readonly LifNeuron[] _neurons;
    private readonly Random _rng;

    // ---- Simulation parameters ----
    public int TrialSteps = 28;          // Simulation steps per stimulus presentation
    public int InputWindow = 20;         // Temporal window where input spikes fall
    public double InputGain = 3.0;       // Input current gain multiplier
    public double ThreshJump = 0.8;      // Lateral threshold jump applied to competing neurons
    public double ThreshDecay = 0.06;    // Threshold decay rate per step
    public int InputJitter = 2;          // Channel temporal jitter (± steps)
    public int ZoneTimeBin = 4;          // Quantization bin width for Zone signatures

    /// <summary>
    /// Input bursts: number of burst volleys per active channel.
    /// Separated by BurstPeriod steps to stagger temporal integration
    /// and eliminate degenerate simultaneous ties.
    /// </summary>
    public int InputBursts = 1;
    public int BurstPeriod = 5;

    /// <summary>
    /// Consecutive steps above threshold required to fire (commitment).
    /// </summary>
    public int CommitSteps = 3;

    /// <summary>
    /// Presence coding: channels below 0.5 do not spike. Silence carries information.
    /// </summary>
    public bool PresenceCoding = false;

    // ---- STDP Plasticity ----
    public bool LearningEnabled = true;
    public double APos = 0.03;           // Potentiation (pre before post)
    public double ANeg = 0.033;          // Depression (post before pre)
    public double TauStdp = 12.0;
    public double WMin = 0.001;
    public double WMax = 0.35;           // Max bound preventing single-channel domination

    public CompetitiveLayer(int neuronCount, int inputCount, int seed)
    {
        NeuronCount = neuronCount;
        InputCount = inputCount;
        _rng = new Random(seed);
        _neurons = new LifNeuron[neuronCount];
        Weights = new double[neuronCount, inputCount];

        for (int i = 0; i < neuronCount; i++)
        {
            _neurons[i] = new LifNeuron();
            double sum = 0;
            for (int d = 0; d < inputCount; d++)
            {
                Weights[i, d] = 0.25 + _rng.NextDouble() * 0.3;
                sum += Weights[i, d];
            }
            for (int d = 0; d < inputCount; d++)
                Weights[i, d] /= sum; // Fair competition: normalized initial weights
        }
    }

    /// <summary>
    /// Presenta un estímulo. Devuelve el orden de primer disparo y la zona
    /// (orden cuantizado en bins). Solo incluye las neuronas que dispararon.
    /// </summary>
    public TrialResult PresentStimulus(double[] x, double noiseAmp)
    {
        // 1) Encoding: latency (default) or presence (binary).
        //    In presence coding, an input channel with value < 0.5 does not spike (preTime = -1).
        int[] preTime = new int[InputCount];
        Array.Fill(preTime, -1);
        for (int d = 0; d < InputCount; d++)
        {
            double v = Math.Clamp(x[d] + (_rng.NextDouble() * 2 - 1) * noiseAmp, 0.0, 1.0);
            if (PresenceCoding && v < 0.5) continue; // channel is silent
            int t = (int)Math.Round((1.0 - v) * InputWindow);
            t += _rng.Next(-InputJitter, InputJitter + 1);
            preTime[d] = Math.Clamp(t, 0, TrialSteps - 1);
        }

        // 2) Input spike agenda per simulation step (with optional bursts).
        var agenda = new List<int>[TrialSteps];
        for (int t = 0; t < TrialSteps; t++) agenda[t] = new List<int>();
        for (int d = 0; d < InputCount; d++)
        {
            if (preTime[d] < 0) continue;
            for (int b = 0; b < InputBursts; b++)
            {
                int tb = preTime[d] + b * BurstPeriod;
                if (tb < TrialSteps) agenda[tb].Add(d);
            }
        }

        foreach (var n in _neurons) { n.Reset(); n.CommitSteps = CommitSteps; }
        int[] firstSpike = new int[NeuronCount];
        Array.Fill(firstSpike, -1);
        int spikesThisTrial = 0;

        // 3) Simulation loop.
        int[] order = new int[NeuronCount];
        for (int i = 0; i < NeuronCount; i++) order[i] = i;

        for (int t = 0; t < TrialSteps; t++)
        {
            // Merit-based evaluation: neurons with highest membrane potential step first.
            Array.Sort(order, (a, b) => _neurons[b].V.CompareTo(_neurons[a].V));

            foreach (int i in order)
            {
                double current = 0;
                foreach (int d in agenda[t])
                    current += Weights[i, d];
                current *= InputGain;

                if (_neurons[i].Step(current))
                {
                    if (firstSpike[i] < 0) firstSpike[i] = t;

                    // Lateral inhibition via threshold jump: firing neuron raises competitors' thresholds.
                    for (int k = 0; k < NeuronCount; k++)
                        if (k != i) _neurons[k].RaiseThreshold(ThreshJump);

                    // STDP with rank discount: earlier firing neurons receive higher reinforcement.
                    if (LearningEnabled) ApplyStdp(i, t, preTime, 1.0 / (1.0 + spikesThisTrial));
                    spikesThisTrial++;
                }
            }

            foreach (var n in _neurons) n.DecayThreshold(ThreshDecay);
        }

        var spikeOrder = Enumerable.Range(0, NeuronCount)
            .Where(i => firstSpike[i] >= 0)
            .OrderBy(i => firstSpike[i])
            .ToList();

        return new TrialResult(spikeOrder, firstSpike, ZoneSignature(firstSpike));
    }

    /// <summary>
    /// Zone signature: neurons binned temporally.
    /// Trials with identical firing patterns within the time tolerance share the same Zone.
    /// </summary>
    private string ZoneSignature(int[] firstSpike)
    {
        var bins = Enumerable.Range(0, NeuronCount)
            .Where(i => firstSpike[i] >= 0)
            .GroupBy(i => firstSpike[i] / ZoneTimeBin)
            .OrderBy(g => g.Key)
            .Select(g => string.Join(",", g.OrderBy(i => i)));
        string sig = string.Join("|", bins);
        return sig.Length == 0 ? "Empty(silence)" : sig;
    }

    private void ApplyStdp(int neuron, int tPost, int[] preTime, double rankDiscount)
    {
        for (int d = 0; d < InputCount; d++)
        {
            if (preTime[d] < 0) continue;
            double dt = tPost - preTime[d];
            double dw = dt >= 0
                ? APos * Math.Exp(-dt / TauStdp)    // Pre before post: potentiation
                : -ANeg * Math.Exp(dt / TauStdp);   // Post before pre: depression
            Weights[neuron, d] = Math.Clamp(Weights[neuron, d] + dw * rankDiscount, WMin, WMax);
        }
        // Weight normalization: keep synaptic resource sum constant
        double sum = 0;
        for (int d = 0; d < InputCount; d++) sum += Weights[neuron, d];
        for (int d = 0; d < InputCount; d++) Weights[neuron, d] /= sum;
    }

    /// <summary>Top input channels preferred by neuron (for inspection).</summary>
    public List<int> TopChannels(int neuron, int count) =>
        Enumerable.Range(0, InputCount)
            .OrderByDescending(d => Weights[neuron, d])
            .Take(count)
            .ToList();
}
