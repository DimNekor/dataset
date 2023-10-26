cd /D D:\Dataset
mkdir D:\Dataset_new
for /r %%a in (*.mp4) do (
	for %%b in ("%%~dpa\.") do (
		set "parent=%%~nxb"
		if %%~zg GTR 464 (
			ffmpeg -i "%%g" -preset ultrafast "D:\Dataset_new\%parent%\%%~ng.mp4"
		)
	)
)