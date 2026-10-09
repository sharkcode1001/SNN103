namespace SNN103;

/// <summary>
/// Neurona integrate-and-fire con fuga (LIF).
/// Integra la corriente de entrada en su potencial de membrana V;
/// cuando V alcanza el umbral, emite un spike y se resetea.
/// Es el "oyente": escucha todas las entradas, solo responde si el
/// estímulo supera lo que ella entiende (su umbral).
/// </summary>
public sealed class LifNeuron
{
    public double V;                    // potencial de membrana actual
    public double VRest = 0.0;          // potencial de reposo
    public double VReset = 0.0;         // potencial tras disparar
    public double Threshold = 1.0;      // umbral actual (sube con la inhibición lateral)
    public double BaseThreshold = 1.0;  // umbral de reposo
    public double TauM = 25.0;          // constante de tiempo de membrana (pasos)
    public int RefractorySteps = 3;     // pasos ciego tras disparar

    /// <summary>
    /// Compromiso: la neurona solo dispara si su potencial se mantiene
    /// sobre el umbral este número de pasos CONSECUTIVOS. Filtra el ruido:
    /// un cruce fugaz por jitter no dispara; solo la evidencia sostenida
    /// ("de verdad entiendo este estímulo") produce un spike.
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
    /// Avanza un paso de simulación. Devuelve true si la neurona disparó
    /// (tras sostener el umbral CommitSteps pasos consecutivos).
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
    /// Inhibición lateral por salto de umbral: cuando una rival dispara,
    /// mi umbral sube y me cuesta más responder. Crea gaps temporales
    /// grandes entre disparos (el jitter no los cruza) y el orden resultante
    /// es un ranking por mérito: la mejor "entendedora" primero.
    /// </summary>
    public void RaiseThreshold(double amount) => Threshold += amount;

    /// <summary>El umbral extra decae hacia el reposo con el tiempo.</summary>
    public void DecayThreshold(double rate) => Threshold += (BaseThreshold - Threshold) * rate;
}
