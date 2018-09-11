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
	$prefix=$_GET['prefix'];
	$learn=$_GET['learn'];
	$beam_size=$_GET['beam_size'];
	$length_norm=$_GET['length_norm'];
	$coverage_norm=$_GET['coverage_norm'];
	$alpha_norm=$_GET['alpha_norm'];

	$url = '127.0.0.1:6542/?source='.urlencode($source).'&prefix='.urlencode($prefix).'&learn='.urlencode($learn).'&beam_size='.urlencode($beam_size).'&length_norm='.urlencode($length_norm).'&coverage_norm='.urlencode($coverage_norm).'&alpha_norm='.urlencode($alpha_norm);
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

