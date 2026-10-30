export class Base {
  save() {
    return this.validate();
  }

  validate() {
    return true;
  }
}

export class User extends Base {
  name = "";

  validate() {
    return this.name.length > 0;
  }

  rename(name) {
    this.name = name;
    return this.save();
  }
}

export class Order {
  parse(text) {
    return text.split(",");
  }
}
