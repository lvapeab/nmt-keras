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
	$source=$_GET['source'];
	$url = '158.42.161.42:6542/?source='.urlencode($source);
	$ch = curl_init();
	curl_setopt($ch, CURLOPT_URL, $url);
	curl_setopt($ch, CURLOPT_RETURNTRANSFER, -1);
	$out = curl_exec($ch);
	curl_close($ch);
	echo $out;
}
else
{
	echo "Server timeout! Try again later.";
}

?>

