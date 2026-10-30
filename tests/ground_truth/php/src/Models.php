<?php

class Base
{
    public function save()
    {
        return $this->validate();
    }

    public function validate()
    {
        return true;
    }
}

class User extends Base
{
    private $name = "";

    public function validate()
    {
        return $this->name !== "";
    }

    public function rename($name)
    {
        $this->name = $name;
        return $this->save();
    }
}

class Order
{
    public function parse($text)
    {
        return explode(",", $text);
    }
}
