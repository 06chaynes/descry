from app.io import Reader, load
from app.models import User


def main():
    user = User()
    user.rename("ann")
    return load(Reader(), "[1]")


def misuse():
    user = User()
    return user.parse("x")
