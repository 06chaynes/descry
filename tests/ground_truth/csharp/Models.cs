public class Base
{
    public virtual bool Save()
    {
        return Validate();
    }

    public virtual bool Validate()
    {
        return true;
    }
}

public class User : Base
{
    private string name = "";

    public override bool Validate()
    {
        return name.Length > 0;
    }

    public bool Rename(string newName)
    {
        name = newName;
        return Save();
    }
}

public class Order
{
    public string[] Parse(string text)
    {
        return text.Split(',');
    }
}
