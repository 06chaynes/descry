require "json"

class Reader
  def parse(text)
    JSON.parse(text)
  end
end

def read_all(text)
  reader = Reader.new
  reader.parse(text)
end

def guess(thing, text)
  thing.parse(text)
end
