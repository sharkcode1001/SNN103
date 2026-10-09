namespace SNN103;

public enum AppLanguage
{
    Spanish,
    English
}

public static class I18n
{
    public static AppLanguage CurrentLanguage { get; set; } = AppLanguage.Spanish;

    public static string T(string es, string en) => CurrentLanguage == AppLanguage.English ? en : es;
}
