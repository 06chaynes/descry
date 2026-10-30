class Base:
    def save(self):
        return self.validate()

    def validate(self):
        return True


class User(Base):
    def validate(self):
        return bool(self.name)

    def rename(self, name):
        self.name = name
        return self.save()


class Order:
    def parse(self, text):
        return text.split(",")
