@echo off
set /p "path=Enter path to Dataset:"

cd /D %path%
set "new_dataset=%CD:~0,2%"
mkdir %new_dataset%\Dataset_new
for /r %%a in (*.mp4) do (
	for %%b in ("%%~dpa\.") do (
		set "parent=%%~nxb"
		if %%~zg GTR 464 (
			ffmpeg -i "%%g" -preset ultrafast "%new_dataset%\Dataset_new\%parent%\%%~ng.mp4"
		)
	)
)