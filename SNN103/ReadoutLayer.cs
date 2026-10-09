namespace SNN103;

/// <summary>
/// Capa de salida SUPERVISADA con REGLA DEL PERCEPTRÓN (Rosenblatt, 1958).
/// Lee el vector binario de disparo de la capa competitiva y aprende el mapeo
/// a las salidas deseadas.
///
/// Por el teorema de convergencia del perceptrón: si los códigos ocultos son
/// linealmente separables para la tarea, esta regla los separa en un número
/// finito de pasos (garantizado). Si no converge, los códigos NO son
/// separables: diagnóstico definitivo, no un problema del optimizador.
///
/// División de trabajo honesta:
///   - La capa competitiva (aleatoria o STDP) expande las entradas a un código,
///   - esta capa aprende la TAREA (entradas -> salidas correctas).
/// </summary>
public sealed class ReadoutLayer
{
    public int InputCount { get; }
    public int OutputCount { get; }

    /// <summary>Tasa de aprendizaje (1.0 es lo estándar para el perceptrón).</summary>
    public double LearningRate = 1.0;

    private readonly double[,] _w; // [salida, entrada]
    private readonly double[] _b;  // sesgos

    public ReadoutLayer(int inputCount, int outputCount, int seed)
    {
        InputCount = inputCount;
        OutputCount = outputCount;
        var rng = new Random(seed);
        _w = new double[outputCount, inputCount];
        _b = new double[outputCount];
        for (int o = 0; o < outputCount; o++)
            for (int i = 0; i < inputCount; i++)
                _w[o, i] = (rng.NextDouble() * 2 - 1) * 0.1;
    }

    private double Raw(int o, double[] x)
    {
        double s = _b[o];
        for (int i = 0; i < InputCount; i++) s += _w[o, i] * x[i];
        return s;
    }

    /// <summary>Predice las salidas (0/1) para un vector de rasgos.</summary>
    public int[] Predict(double[] x)
    {
        var y = new int[OutputCount];
        for (int o = 0; o < OutputCount; o++)
            y[o] = Raw(o, x) >= 0 ? 1 : 0;
        return y;
    }

    /// <summary>La clase con mayor activación (argmax). Nunca se abstiene.</summary>
    public int PredictBest(double[] x)
    {
        int best = 0;
        double bestV = Raw(0, x);
        for (int o = 1; o < OutputCount; o++)
        {
            double v = Raw(o, x);
            if (v > bestV) { bestV = v; best = o; }
        }
        return best;
    }

    /// <summary>
    /// Un paso de la regla del perceptrón. Solo corrige cuando se equivoca.
    /// Devuelve 1 si hubo error en alguna salida, 0 si todo correcto.
    /// </summary>
    public double Train(double[] x, int[] target)
    {
        double err = 0;
        for (int o = 0; o < OutputCount; o++)
        {
            int t = target[o] == 1 ? 1 : -1;   // 0/1 -> -1/+1
            int y = Raw(o, x) >= 0 ? 1 : -1;
            if (y != t)
            {
                for (int i = 0; i < InputCount; i++)
                    _w[o, i] += LearningRate * t * x[i];
                _b[o] += LearningRate * t;
                err += 1;
            }
        }
        return err;
    }
}
