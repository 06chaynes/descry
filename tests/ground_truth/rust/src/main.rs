mod io;
mod models;

use io::{load, Reader};
use models::User;

fn main() {
    let mut user = User::new();
    user.rename("ann");
    let _ = load(&Reader, "[1]");
}
