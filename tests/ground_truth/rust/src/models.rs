pub trait Save {
    fn validate(&self) -> bool;

    fn save(&self) -> bool {
        self.validate()
    }
}

pub struct User {
    name: String,
}

impl User {
    pub fn new() -> Self {
        User { name: String::new() }
    }

    pub fn rename(&mut self, name: &str) -> bool {
        self.name = name.to_string();
        self.save()
    }
}

impl Save for User {
    fn validate(&self) -> bool {
        !self.name.is_empty()
    }
}

pub struct Order;

impl Order {
    pub fn parse(&self, text: &str) -> Vec<String> {
        text.split(',').map(String::from).collect()
    }
}

impl Save for Order {
    fn validate(&self) -> bool {
        true
    }
}
