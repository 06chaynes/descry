public class Reader
{
    public int Parse(string text)
    {
        return text.Length;
    }
}

public static class Io
{
    public static int Load(Reader reader, string text)
    {
        return reader.Parse(text);
    }
}
