export class Reader {
  parse(text) {
    return JSON.parse(text);
  }
}

export function load(text) {
  const reader = new Reader();
  return reader.parse(text);
}

export function guess(thing, text) {
  return thing.parse(text);
}
