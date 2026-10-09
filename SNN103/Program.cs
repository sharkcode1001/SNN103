using System;
using SNN103;

Console.OutputEncoding = System.Text.Encoding.UTF8;

// CLI argument parsing
for (int i = 0; i < args.Length; i++)
{
    string a = args[i].ToLowerInvariant();
    if (a == "--lang" && i + 1 < args.Length)
    {
        string l = args[i + 1].ToLowerInvariant();
        if (l == "en" || l == "english") I18n.CurrentLanguage = AppLanguage.English;
        else if (l == "es" || l == "spanish") I18n.CurrentLanguage = AppLanguage.Spanish;
    }
    else if (a == "--en")
    {
        I18n.CurrentLanguage = AppLanguage.English;
    }
    else if (a == "--es")
    {
        I18n.CurrentLanguage = AppLanguage.Spanish;
    }
}

if (args.Length > 0 && !args[0].StartsWith("--lang") && args[0] != "--en" && args[0] != "--es")
{
    string cmd = args[0].ToLowerInvariant();
    switch (cmd)
    {
        case "--benchmark":
        case "-b":
        case "benchmark":
        case "1":
            Experimento.Run10();
            return;

        case "--pilot":
        case "-p":
        case "pilot":
        case "2":
            Experimento.Run();
            return;

        case "--noise":
        case "-n":
        case "noise":
            Experimento.RunNoiseRobustnessDemo();
            return;

        case "--all":
        case "-a":
        case "all":
            RunFullSuite();
            return;

        case "--csv":
        case "-c":
            string csvPath = args.Length > 1 ? args[1] : "4_in_2_out.csv";
            CsvExperiment.Run(csvPath);
            return;

        case "--help":
        case "-h":
            PrintHelp();
            return;

        default:
            CsvExperiment.Run(args[0]);
            return;
    }
}

while (true)
{
    Console.WriteLine();
    Console.ForegroundColor = ConsoleColor.Cyan;
    Console.WriteLine("========================================================================");
    Console.WriteLine(" SNN103: Neuromorphic Rank-Order Spiking Neural Network Suite");
    Console.WriteLine(I18n.T(
        " Plataforma de Investigación (.NET 10) | Clasificación Guiada por Eventos",
        " Research Platform (.NET 10) | Event-Driven Digit Classification"));
    Console.WriteLine("========================================================================");
    Console.ResetColor();

    string currentLangText = I18n.CurrentLanguage == AppLanguage.English ? "English [EN]" : "Español [ES]";
    Console.ForegroundColor = ConsoleColor.DarkGray;
    Console.WriteLine(I18n.T($"  Idioma actual: {currentLangText}", $"  Current language: {currentLangText}"));
    Console.ResetColor();
    Console.WriteLine();

    Console.WriteLine(I18n.T(
        "  [1] Experimento Central (Secc. 4.5): 10 Clases × 5 Semillas\n      -> Ablación: Orden (1/rank) vs Binario vs MLP clásico.",
        "  [1] Core Experiment (Sec. 4.5): 10 Classes × 5 Seeds\n      -> Ablation: Order (1/rank) vs Binary vs Classic MLP."));
    Console.WriteLine(I18n.T(
        "  [2] Experimento Piloto (Secc. 4.5): 4 Clases (Visualización)\n      -> Matriz de confusión, raster plot y balance eventos/multiplicaciones.",
        "  [2] Pilot Experiment (Sec. 4.5): 4 Classes (Visualization)\n      -> Confusion matrix, raster plot and events/multiplications budget."));
    Console.WriteLine(I18n.T(
        "  [3] Demostración Teórica (Secc. 4.2): Robustez ante Ruido (2x4 vs 1x8)\n      -> Evidencia empírica de por qué subregiones de 4 salvan el código.",
        "  [3] Theoretical Demo (Sec. 4.2): Noise Robustness (2x4 vs 1x8)\n      -> Empirical proof of why 4-neuron subregions preserve the code."));
    Console.WriteLine(I18n.T(
        "  [4] Ejecutar Suite Completa y Generar Todas las Figuras",
        "  [4] Run Full Suite and Export All Figures"));
    Console.WriteLine(I18n.T(
        "  [5] Evaluar Archivo CSV Personalizado",
        "  [5] Evaluate Custom CSV File"));
    Console.WriteLine(I18n.T(
        "  [L] Cambiar Idioma / Switch Language (Español / English)",
        "  [L] Switch Language / Cambiar Idioma (English / Español)"));
    Console.WriteLine(I18n.T(
        "  [?] Fundamentos del Paper (¿Por qué el código de orden funciona?)",
        "  [?] Paper Fundamentals (Why rank-order coding works)"));
    Console.WriteLine(I18n.T(
        "  [0] Salir",
        "  [0] Exit"));
    Console.WriteLine();
    Console.Write(I18n.T("Seleccione una opción: ", "Select an option: "));
    
    string? choice = Console.ReadLine()?.Trim();
    if (string.IsNullOrEmpty(choice) || choice == "0") break;
    Console.WriteLine();

    switch (choice.ToLowerInvariant())
    {
        case "1":
            Experimento.Run10();
            break;
        case "2":
            Experimento.Run();
            break;
        case "3":
            Experimento.RunNoiseRobustnessDemo();
            break;
        case "4":
            RunFullSuite();
            break;
        case "5":
            Console.Write(I18n.T("Ruta del archivo CSV (Enter = 4_in_2_out.csv): ", "CSV file path (Enter = 4_in_2_out.csv): "));
            string csv = Console.ReadLine()?.Trim() ?? "";
            if (csv.Length == 0) csv = "4_in_2_out.csv";
            CsvExperiment.Run(csv);
            break;
        case "l":
            I18n.CurrentLanguage = I18n.CurrentLanguage == AppLanguage.Spanish ? AppLanguage.English : AppLanguage.Spanish;
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine(I18n.T("Idioma cambiado a Español.", "Language switched to English."));
            Console.ResetColor();
            break;
        case "?":
        case "help":
            PrintTheorySummary();
            break;
        default:
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine(I18n.T("Opción no válida.", "Invalid option."));
            Console.ResetColor();
            break;
    }
}

static void RunFullSuite()
{
    Console.ForegroundColor = ConsoleColor.Green;
    Console.WriteLine(I18n.T(">>> Ejecutando Suite Experimental Completa (Paper SNN103)...", ">>> Running Complete Experimental Suite (SNN103 Paper)..."));
    Console.ResetColor();
    Console.WriteLine();
    
    Console.WriteLine(I18n.T("=== ETAPA 1/3: Demostración de Robustez ante Ruido (Sección 4.2) ===", "=== STAGE 1/3: Noise Robustness Demo (Section 4.2) ==="));
    Experimento.RunNoiseRobustnessDemo();
    Console.WriteLine();

    Console.WriteLine(I18n.T("=== ETAPA 2/3: Piloto Visual (4 Clases) ===", "=== STAGE 2/3: Visual Pilot (4 Classes) ==="));
    Experimento.Run();
    Console.WriteLine();

    Console.WriteLine(I18n.T("=== ETAPA 3/3: Validación Estadística Central (10 Clases × 5 Semillas) ===", "=== STAGE 3/3: Central Statistical Validation (10 Classes × 5 Seeds) ==="));
    Experimento.Run10();
    Console.WriteLine();

    Console.ForegroundColor = ConsoleColor.Green;
    Console.WriteLine(I18n.T(">>> Suite completada. Figuras generadas en 'plots/'.", ">>> Suite completed. Figures exported to 'plots/'."));
    Console.ResetColor();
}

static void PrintTheorySummary()
{
    Console.ForegroundColor = ConsoleColor.Cyan;
    Console.WriteLine("========================================================================");
    Console.WriteLine(I18n.T(" FUNDAMENTOS TEÓRICOS Y ARQUITECTURA DE SNN103", " THEORETICAL FOUNDATIONS AND ARCHITECTURE OF SNN103"));
    Console.WriteLine("========================================================================");
    Console.ResetColor();
    Console.WriteLine(I18n.T(
        " 1. EL CÓDIGO DE LÍNEA B103 Y EL ORDEN DE PRIMER DISPARO:\n" +
        "    En n canales paralelos, el orden de llegada codifica n! combinaciones.\n" +
        "    En lugar de solo medir si una neurona disparó (1 bit), medimos en qué\n" +
        "    puesto llegó (1/rango). El orden aporta ~17 puntos sobre el código binario.\n",
        " 1. THE B103 LINE CODE AND FIRST-SPIKE ORDER:\n" +
        "    Across n parallel channels, arrival order encodes n! distinct states.\n" +
        "    Instead of observing only whether a neuron fired (1 bit), we encode\n" +
        "    its arrival position (1/rank). Order yields ~17 points over binary features.\n"));
    Console.WriteLine(I18n.T(
        " 2. POR QUÉ SUBREGIONES DE 4 NEURONAS (Sección 4.2):\n" +
        "    Bajo ruido/jitter temporal, un grupo de 8 neuronas colapsa (62.3% precisión),\n" +
        "    mientras que subregiones modulares de 4 retienen el 98.9% de exactitud.\n",
        " 2. WHY 4-NEURON SUBREGIONS (Section 4.2):\n" +
        "    Under temporal jitter, an 8-neuron group degrades (62.3% exact orders),\n" +
        "    whereas modular 4-neuron subregions retain 98.9% accuracy.\n"));
    Console.WriteLine(I18n.T(
        " 3. EFICIENCIA COMPUTACIONAL BASADA EN EVENTOS:\n" +
        "    La red procesa mediante ráfagas e inhibición lateral en ~169 spikes por\n" +
        "    decisión, frente a 2368 multiplicaciones de un MLP clásico (~14x menos eventos).\n",
        " 3. EVENT-DRIVEN COMPUTATIONAL EFFICIENCY:\n" +
        "    Processing relies on bursts and lateral inhibition in ~169 spikes per\n" +
        "    decision, compared to 2368 multiplications in a classic MLP (~14x fewer events).\n"));
    Console.WriteLine(I18n.T(
        " 4. LÍMITE RECONOCIDO EN EL PAPER:\n" +
        "    El front-end actual usa pesos aleatorios fijos (expansión biológica estable).\n" +
        "    El techo de precisión frente al MLP proviene de que la capa fija no aprende\n" +
        "    estadísticamente; más grupos aleatorios no cierran la brecha (criterio STOP).\n",
        " 4. ACKNOWLEDGED ARCHITECTURAL CEILING:\n" +
        "    The front-end uses fixed random weights (stable biological expansion).\n" +
        "    The remaining gap to the MLP is due to the non-adaptive front-end;\n" +
        "    increasing random groups does not close the gap (pre-registered STOP criterion).\n"));
    Console.WriteLine("========================================================================");
}

static void PrintHelp()
{
    Console.WriteLine(I18n.T("Uso: SNN103 [opciones]", "Usage: SNN103 [options]"));
    Console.WriteLine();
    Console.WriteLine(I18n.T("Opciones CLI:", "CLI Options:"));
    Console.WriteLine(I18n.T("  --benchmark, -b     Ejecuta la ablación central de 10 clases (5 semillas).", "  --benchmark, -b     Run core 10-class ablation (5 seeds)."));
    Console.WriteLine(I18n.T("  --pilot, -p         Ejecuta el experimento visual de 4 clases con raster plot.", "  --pilot, -p         Run visual 4-class experiment with raster plot."));
    Console.WriteLine(I18n.T("  --noise, -n         Ejecuta la prueba de robustez de subregiones ante ruido.", "  --noise, -n         Run subregions noise robustness demo."));
    Console.WriteLine(I18n.T("  --all, -a           Ejecuta todas las validaciones y genera el set de figuras.", "  --all, -a           Run all validations and generate figures."));
    Console.WriteLine(I18n.T("  --csv <ruta>, -c    Evalúa un archivo CSV personalizado.", "  --csv <path>, -c    Evaluate a custom CSV file."));
    Console.WriteLine(I18n.T("  --lang <es|en>      Establece el idioma (es o en).", "  --lang <es|en>      Set language (es or en)."));
    Console.WriteLine(I18n.T("  --help, -h          Muestra este mensaje de ayuda.", "  --help, -h          Show this help message."));
}
