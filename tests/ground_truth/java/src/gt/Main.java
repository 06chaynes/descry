package gt;

public class Main {
    public static void main(String[] args) {
        User user = new User();
        user.rename("ann");
        Io.load(new Reader(), "[1]");
    }
}
