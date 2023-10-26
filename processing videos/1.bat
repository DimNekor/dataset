@echo off
set /p "path=Enter path to Dataset:"

cd /D %path%
set "new_dataset=%CD:~0,2%"
if not exist "%new_dataset%\Dataset_new" mkdir %new_dataset%\Dataset_new
for /r %%a in (*.mp4) do (
	for %%b in ("%%~dpa\.") do (
		set "parent=%%~nxb"
		if %%~za GTR 464 (
			if not exist ""%new_dataset%\Dataset_new\%parent%\" mkdir "%new_dataset%\Dataset_new\%parent%\
			ffmpeg -i "%%a" -preset ultrafast "%new_dataset%\Dataset_new\%parent%\%%~na.mp4"
		)
	)
)