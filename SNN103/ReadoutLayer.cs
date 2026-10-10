namespace SNN103;

/// <summary>
/// Supervised output layer using the classic Perceptron learning rule (Rosenblatt, 1958).
/// Reads feature vectors from the competitive front-end and learns the mapping
/// to desired target outputs.
///
/// Perceptron convergence theorem: if feature representations are linearly
/// separable, this rule converges in a finite number of steps.
/// </summary>
public sealed class ReadoutLayer
{
    public int InputCount { get; }
    public int OutputCount { get; }

    /// <summary>Learning rate (1.0 standard for linear perceptron).</summary>
    public double LearningRate = 1.0;

    private readonly double[,] _w; // [output, input]
    private readonly double[] _b;  // biases

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

    /// <summary>Predicts binary outputs (0/1) for a given feature vector.</summary>
    public int[] Predict(double[] x)
    {
        var y = new int[OutputCount];
        for (int o = 0; o < OutputCount; o++)
            y[o] = Raw(o, x) >= 0 ? 1 : 0;
        return y;
    }

    /// <summary>Returns the class index with highest activation (argmax).</summary>
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
    /// Single update step using the Perceptron learning rule. Updates weights on misclassification.
    /// Returns 1.0 if any output was misclassified, 0.0 otherwise.
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
