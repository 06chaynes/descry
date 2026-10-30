pub struct Reader;

impl Reader {
    pub fn parse(&self, text: &str) -> usize {
        text.len()
    }
}

pub fn load(reader: &Reader, text: &str) -> usize {
    reader.parse(text)
}
