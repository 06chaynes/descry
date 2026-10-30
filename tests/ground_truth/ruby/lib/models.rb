class Base
  def save
    validate
  end

  def validate
    true
  end
end

class User < Base
  def validate
    !@name.to_s.empty?
  end

  def rename(name)
    @name = name
    save
  end
end

class Order
  def parse(text)
    text.split(",")
  end
end
