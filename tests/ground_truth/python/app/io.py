import json


class Reader:
    def parse(self, text):
        return json.loads(text)


def load(reader: Reader, text):
    return reader.parse(text)


def guess(thing, text):
    return thing.parse(text)
