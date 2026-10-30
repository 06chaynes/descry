import 'dart:convert';

class Reader {
  Object? parse(String text) => jsonDecode(text);
}

Object? load(Reader reader, String text) => reader.parse(text);

Object? guess(dynamic thing, String text) => thing.parse(text);
