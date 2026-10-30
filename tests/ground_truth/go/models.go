package main

import "strings"

type User struct{ name string }

func NewUser() *User { return &User{} }

func (u *User) Rename(name string) bool {
	u.name = name
	return u.Save()
}

func (u *User) Save() bool { return u.Validate() }

func (u *User) Validate() bool { return u.name != "" }

type Order struct{}

func (o Order) Parse(text string) []string { return strings.Split(text, ",") }
