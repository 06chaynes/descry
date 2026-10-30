<?php

require_once __DIR__ . "/Io.php";
require_once __DIR__ . "/Models.php";

function main()
{
    $user = new User();
    $user->rename("ann");
    return load(new Reader(), "[1]");
}

function misuse()
{
    $user = new User();
    return $user->parse("x");
}
