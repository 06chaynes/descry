<?php

class Reader
{
    public function parse($text)
    {
        return json_decode($text);
    }
}

function load(Reader $reader, $text)
{
    return $reader->parse($text);
}

function guess($thing, $text)
{
    return $thing->parse($text);
}
