# finsta360
finsta360 - python script to finalize incomplete MP4 of Insta360 ONE X

# What is this?
If you have Insta360 [ONE X](https://www.insta360.com/product/insta360-onex)
and you have some corrupted files, this is for you!

Action cams sometimes fail to save movie file and players couldn't play it but the DATA is there!
There are several softwares like
* [Untrunc](https://github.com/ponchio/untrunc)
* [movrepair](https://github.com/NiklasRosenstein/movrepair)

but they didn't work for my case.
So I wrote this python script from scratch.
And it works! at least for my case.

I hope this would help some people suffering from the same situation.

# How does it work?
From (only and possibly incomplete) `mdat` data, this script regenerates the sample tables for H264 and AAC streams, and reconstructs `moov` with the help of reference `moov` from the complete MP4 from the same camera.

Note that this script is only for Insta360 [ONE X](https://www.insta360.com/product/insta360-onex), but the technique is applicable to MP4 files with H264/AAC.

# How to use?
All you need is python.
I only use `tqdm` for checking the progress.
Other than that, you need nothing.

Like other recovery software, `finsta360` requires one complete MP4 file as a reference in addition to your corrupted MP4 file.
If you have the files like
* complete file: `../Data/MP4/VID_20191023_195632_00_004.insv`
* corrupted file: `../Data/MP4/VID_20191023_202638_00_005.insv`

<pre>
$ ./finsta360 -s ../Data/MP4/VID_20191023_202638_00_005.insv\
 -r ../Data/MP4/VID_20191023_195632_00_004.insv\
 -o finsta360_00_005.insv
</pre>
this would give you recovered file `finsta360_00_005.insv`.

## Help
<pre>
$ ./finsta360 -h
finsta360.py : to finalize incomplete MP4 of Insta360 ONE-X
https://github.com/kichiki/finsta360
USAGE: finsta360.py [options]
	-s file : source file, that is, corrupted mp4 (insv) file
	-r file : complete mp4 (insv) file as a reference
	-o file : output recovered mp4 (insv) file
	-v      : to set verbose mode
	-k      : to keep temporary files
	          (reference and recovered moov files, finsta360*.moov)
If you provide only source file (-s), program prints the metadata
If you dont provide output file (-o), program just runs without writing
</pre>

# Jupyter notebook
I developed this script on Jupyter notebook.
For those who want to play interatively on Jupyter notebook, I put it [here](finsta360.ipynb).

# License
This software is released under the GNU General Public License v3.0, see [LICENSE](LICENSE).

# Author
* [Kengo Ichiki](https://github.com/kichiki)