namespace SNN103;

/// <summary>
/// Leaky Integrate-and-Fire (LIF) neuron model.
/// Integrates input current into membrane potential V;
/// when V reaches threshold, emits a spike and resets.
/// Responds only when the stimulus drives potential above its threshold.
/// </summary>
public sealed class LifNeuron
{
    public double V;                    // Current membrane potential
    public double VRest = 0.0;          // Resting potential
    public double VReset = 0.0;         // Post-spike reset potential
    public double Threshold = 1.0;      // Dynamic threshold (increases with lateral inhibition)
    public double BaseThreshold = 1.0;  // Baseline resting threshold
    public double TauM = 25.0;          // Membrane time constant (steps)
    public int RefractorySteps = 3;     // Refractory period steps after firing

    /// <summary>
    /// Commitment steps: neuron only fires if potential remains
    /// above threshold for this number of CONSECUTIVE steps.
    /// Filters out transient high-frequency jitter.
    /// </summary>
    public int CommitSteps = 3;

    private int _refractoryLeft;
    private int _aboveCount;

    public void Reset()
    {
        V = VRest;
        Threshold = BaseThreshold;
        _refractoryLeft = 0;
        _aboveCount = 0;
    }

    /// <summary>
    /// Advances simulation by one step. Returns true if the neuron fired
    /// (after sustaining potential above threshold for CommitSteps consecutive steps).
    /// </summary>
    public bool Step(double inputCurrent)
    {
        if (_refractoryLeft > 0)
        {
            _refractoryLeft--;
            return false;
        }

        V += (-(V - VRest) / TauM) + inputCurrent;

        if (V >= Threshold)
        {
            if (++_aboveCount >= CommitSteps)
            {
                V = VReset;
                _refractoryLeft = RefractorySteps;
                _aboveCount = 0;
                return true;
            }
        }
        else
        {
            _aboveCount = 0;
        }
        return false;
    }

    /// <summary>
    /// Lateral inhibition via threshold jump: when a competing neuron fires,
    /// threshold increases to enforce temporal gaps and clean rank separation.
    /// </summary>
    public void RaiseThreshold(double amount) => Threshold += amount;

    /// <summary>Threshold decays back toward base resting value over time.</summary>
    public void DecayThreshold(double rate) => Threshold += (BaseThreshold - Threshold) * rate;
}
