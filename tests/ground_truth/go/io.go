package main

import "encoding/json"

type Reader struct{}

func (r Reader) Parse(text string) any {
	var v any
	_ = json.Unmarshal([]byte(text), &v)
	return v
}

func Load(reader Reader, text string) any { return reader.Parse(text) }
