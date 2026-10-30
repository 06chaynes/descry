export class Reader {
  parse(text: string): unknown {
    return JSON.parse(text);
  }
}

export function load(reader: Reader, text: string): unknown {
  return reader.parse(text);
}

export function guess(thing: any, text: string): unknown {
  return thing.parse(text);
}
