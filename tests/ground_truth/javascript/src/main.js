import { load } from "./io.js";
import { User } from "./models.js";

export function main() {
  const user = new User();
  user.rename("ann");
  return load("[1]");
}

export function misuse() {
  const user = new User();
  return user.parse("x");
}
