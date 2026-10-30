export class Base {
  save(): boolean {
    return this.validate();
  }

  validate(): boolean {
    return true;
  }
}

export class User extends Base {
  name = "";

  validate(): boolean {
    return this.name.length > 0;
  }

  rename(name: string): boolean {
    this.name = name;
    return this.save();
  }
}

export class Order {
  parse(text: string): string[] {
    return text.split(",");
  }
}
