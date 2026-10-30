require_relative "io"
require_relative "models"

def main
  user = User.new
  user.rename("ann")
  read_all("[1]")
end

def misuse
  user = User.new
  user.parse("x")
end
