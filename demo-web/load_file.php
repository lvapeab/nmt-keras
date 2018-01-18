<?php

header('Content-Type: text/plain; charset=utf-8');

try {

    // Undefined | Multiple Files | $_FILES Corruption Attack
    // If this request falls under any of them, treat it invalid.
    if (
        !isset($_FILES['source_file']['error']) ||
        is_array($_FILES['source_file']['error'])
    ) {
        throw new RuntimeException('Invalid parameters.');
    }

    // Check $_FILES['source_file']['error'] value.
    switch ($_FILES['source_file']['error']) {
        case UPLOAD_ERR_OK:
            break;
        case UPLOAD_ERR_NO_FILE:
            throw new RuntimeException('No file sent.');
        case UPLOAD_ERR_INI_SIZE:
        case UPLOAD_ERR_FORM_SIZE:
            throw new RuntimeException('Exceeded filesize limit.');
        default:
            throw new RuntimeException('Unknown errors.');
    }

    // You should also check filesize here.
    if ($_FILES['source_file']['size'] > 1000000) {
        throw new RuntimeException('Exceeded filesize limit.');
    }


 <?php
$myfile = fopen($_FILES['source_file']['name'], "r") or die("Unable to open file!");
echo mb_convert_encoding(fread($myfile,filesize($_FILES['source_file']['name'])), "UTF-8");

fclose($myfile);
?>




} catch (RuntimeException $e) {

    echo $e->getMessage();

}

?>