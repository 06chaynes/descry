class Base {
  bool save() => validate();

  bool validate() => true;
}

class User extends Base {
  String name = '';

  @override
  bool validate() => name.isNotEmpty;

  bool rename(String newName) {
    name = newName;
    return save();
  }
}

class Order {
  List<String> parse(String text) => text.split(',');
}
