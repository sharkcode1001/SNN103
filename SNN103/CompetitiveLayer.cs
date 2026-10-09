namespace SNN103;

/// <summary>
/// Resultado de presentar un estímulo: orden de primer disparo,
/// tiempos y la "zona" (orden cuantizado en bins temporales para
/// tolerancia al jitter, como toda zona de un código de línea).
/// </summary>
public sealed record TrialResult(List<int> SpikeOrder, int[] FirstSpikeTimes, string Zone);

/// <summary>
/// Capa competitiva de neuronas de pulsos. Todas las neuronas escuchan
/// todas las entradas; compiten por inhibición lateral y solo las que
/// "entienden" el estímulo responden. El ORDEN en que disparan es la
/// representación interna (las "zonas" del B103, pero emergentes, sin
/// tablas externas).
///
/// Codificación de entrada: latencia (time-to-first-spike) por defecto; cada canal
/// emite un único spike, cuanto más intenso el estímulo más temprano.
/// Con PresenceCoding = true (entradas binarias), un canal en 0 no emite spike.
/// Aprendizaje: STDP clásico por pares con descuento por rango (Hebbiano temporal).
/// </summary>
public sealed class CompetitiveLayer
{
    public int NeuronCount { get; }
    public int InputCount { get; }

    /// <summary>Pesos sinápticos [neurona, canal]. Cada fila suma 1.</summary>
    public double[,] Weights { get; }

    private readonly LifNeuron[] _neurons;
    private readonly Random _rng;

    // ---- Parámetros de simulación (ajustables) ----
    public int TrialSteps = 28;          // pasos por presentación (solo la primera oleada)
    public int InputWindow = 20;         // ventana donde caen los spikes de entrada
    public double InputGain = 3.0;       // escala de la corriente de entrada
    public double ThreshJump = 0.8;      // cuánto sube el umbral ajeno al disparar
    public double ThreshDecay = 0.06;    // qué tan rápido decae ese umbral extra
    public int InputJitter = 2;          // jitter temporal del canal (± pasos)
    public int ZoneTimeBin = 4;          // tolerancia temporal de la zona (pasos)

    /// <summary>
    /// Ráfagas de entrada: cada canal que habla emite este número de volleys,
    /// separados por BurstPeriod pasos. Con 1 (defecto) es el comportamiento
    /// clásico de un solo spike por canal. Con más de 1, las neuronas integran
    /// en el tiempo y cruzan el umbral en momentos DISTINTOS según su drive:
    /// así el orden de disparo es un ranking real, no un empate.
    /// </summary>
    public int InputBursts = 1;
    public int BurstPeriod = 5;

    /// <summary>
    /// Pasos consecutivos sobre el umbral para disparar (compromiso).
    /// Se aplica a las neuronas al presentar. Más alto = más separación
    /// temporal entre disparos del mismo volley (menos empates).
    /// </summary>
    public int CommitSteps = 3;

    /// <summary>
    /// Codificación por presencia (para entradas binarias): un canal con valor
    /// &lt; 0.5 no emite spike en el ensayo (preTime = -1). El silencio también
    /// es información. Si es false, codificación por latencia (todos los canales
    /// emiten un spike, antes o después según su valor).
    /// </summary>
    public bool PresenceCoding = false;

    // ---- STDP ----
    public bool LearningEnabled = true;
    public double APos = 0.03;           // potenciación (pre antes que post)
    public double ANeg = 0.033;          // depresión (pre después que post)
    public double TauStdp = 12.0;
    public double WMin = 0.001;
    public double WMax = 0.35; // tope: impide que un solo canal acapare a la neurona
                               // -> detectores conjuntivos (combinaciones), no monocanal

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
                Weights[i, d] /= sum; // competencia justa: todas parten iguales
        }
    }

    /// <summary>
    /// Presenta un estímulo. Devuelve el orden de primer disparo y la zona
    /// (orden cuantizado en bins). Solo incluye las neuronas que dispararon.
    /// </summary>
    public TrialResult PresentStimulus(double[] x, double noiseAmp)
    {
        // 1) Codificación: latencia (defecto) o presencia (binario).
        //    En presencia, un canal con valor < 0.5 no emite spike (preTime = -1).
        int[] preTime = new int[InputCount];
        Array.Fill(preTime, -1);
        for (int d = 0; d < InputCount; d++)
        {
            double v = Math.Clamp(x[d] + (_rng.NextDouble() * 2 - 1) * noiseAmp, 0.0, 1.0);
            if (PresenceCoding && v < 0.5) continue; // este canal no habla en este ensayo
            int t = (int)Math.Round((1.0 - v) * InputWindow);
            t += _rng.Next(-InputJitter, InputJitter + 1);
            preTime[d] = Math.Clamp(t, 0, TrialSteps - 1);
        }

        // 2) Agenda de spikes de entrada por paso (con ráfagas opcionales).
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

        // 3) Simulación.
        int[] order = new int[NeuronCount];
        for (int i = 0; i < NeuronCount; i++) order[i] = i;

        for (int t = 0; t < TrialSteps; t++)
        {
            // Evaluación por mérito: la neurona con mayor potencial va primero.
            // Determinista y con sentido: la más "cargada" (mejor match) dispara antes.
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

                    // Inhibición lateral por salto de umbral: quien dispara
                    // encarece la respuesta de las demás -> gaps temporales
                    // grandes y orden = ranking por calidad de match.
                    for (int k = 0; k < NeuronCount; k++)
                        if (k != i) _neurons[k].RaiseThreshold(ThreshJump);

                    // STDP con descuento por rango: el que dispara primero es
                    // el más causal con el estímulo, luego aprende más.
                    // Esto diferencia a los ganadores por bandas temporales.
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
    /// La zona: neuronas agrupadas por bin temporal. Dos ensayos con el mismo
    /// patrón de disparo dentro de la tolerancia caen en la misma zona.
    /// </summary>
    private string ZoneSignature(int[] firstSpike)
    {
        var bins = Enumerable.Range(0, NeuronCount)
            .Where(i => firstSpike[i] >= 0)
            .GroupBy(i => firstSpike[i] / ZoneTimeBin)
            .OrderBy(g => g.Key)
            .Select(g => string.Join(",", g.OrderBy(i => i)));
        string sig = string.Join("|", bins);
        return sig.Length == 0 ? "∅(silencio)" : sig;
    }

    private void ApplyStdp(int neuron, int tPost, int[] preTime, double rankDiscount)
    {
        for (int d = 0; d < InputCount; d++)
        {
            if (preTime[d] < 0) continue; // este canal no habló: nada que aprender de él
            double dt = tPost - preTime[d];
            double dw = dt >= 0
                ? APos * Math.Exp(-dt / TauStdp)    // pre causó post: refuerzo
                : -ANeg * Math.Exp(dt / TauStdp);   // post sin pre: olvido
            Weights[neuron, d] = Math.Clamp(Weights[neuron, d] + dw * rankDiscount, WMin, WMax);
        }
        // Normalización: la neurona reparte una cantidad fija de "atención".
        double sum = 0;
        for (int d = 0; d < InputCount; d++) sum += Weights[neuron, d];
        for (int d = 0; d < InputCount; d++) Weights[neuron, d] /= sum;
    }

    /// <summary>Canales que más "entiende" cada neurona (para inspeccionar especialización).</summary>
    public List<int> TopChannels(int neuron, int count) =>
        Enumerable.Range(0, InputCount)
            .OrderByDescending(d => Weights[neuron, d])
            .Take(count)
            .ToList();
}
