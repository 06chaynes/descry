import { load, Reader } from "./io";
import { User } from "./models";

export function main(): unknown {
  const user = new User();
  user.rename("ann");
  return load(new Reader(), "[1]");
}

export function misuse(): unknown {
  const user = new User();
  // @ts-expect-error: User has no parse
  return user.parse("x");
}
