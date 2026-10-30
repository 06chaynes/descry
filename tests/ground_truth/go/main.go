package main

func main() {
	user := NewUser()
	user.Rename("ann")
	_ = Load(Reader{}, "[1]")
}
