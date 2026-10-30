public static class Program
{
    public static void Main()
    {
        var user = new User();
        user.Rename("ann");
        Io.Load(new Reader(), "[1]");
    }
}
