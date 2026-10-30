package gt;

public class User extends Base {
    private String name = "";

    @Override
    public boolean validate() {
        return !name.isEmpty();
    }

    public boolean rename(String name) {
        this.name = name;
        return save();
    }
}
