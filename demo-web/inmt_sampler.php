<?php

# make sure that only one script at a time accesses the sampling server
$fp = fopen('/tmp/flock', 'w');

function file_get_contents_utf8($fn) {
     $content = file_get_contents($fn);
      return mb_convert_encoding($content, 'UTF-8',
          mb_detect_encoding($content, 'UTF-8, ISO-8859-1', true));
}


if (flock($fp, LOCK_EX))
{
	$source=$_GET['source']; $prefix=$_GET['prefix']; $learn=$_GET['learn'];
	$url = 'http://localhost:8888/?source='.urlencode($source).'&prefix='.urlencode($prefix).'&learn='.urlencode($learn);
    $out = file_get_contents($url);
	echo $out;
}
else
{
	echo "Server timeout! Try again later.";
}

?>

