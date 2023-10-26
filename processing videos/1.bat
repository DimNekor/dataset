cd /D D:\Dataset
mkdir D:\Dataset_new
for /r %%g in (*.mp4) do (
	if %%~zg GTR 464 (
	ffmpeg -i "%%g" -preset ultrafast "D:\Dataset_new\%%~ng.mp4"
	)
)