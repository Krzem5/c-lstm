@echo off
cls
set _INCLUDE=%INCLUDE%
set INCLUDE=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\include;..\src\include;%INCLUDE%
if exist build rmdir /s /q build
mkdir build
cd build
if %1.==. goto dbg
if %1==-r (
	cl /c /permissive- /GS /W3 /Zc:wchar_t /Gm- /sdl /Zc:inline /fp:precise /D "NDEBUG" /D "_WINDOWS" /D "_UNICODE" /D "UNICODE" /errorReport:none /WX /Zc:forScope /Gd /Oi /FC /EHsc /nologo /diagnostics:column /GL /Gy /Zi /O2 /Oi /MD ../src/main.c ../src/lstm/*.c&&link *.obj /OUT:lstm.exe /DYNAMICBASE "kernel32.lib" "user32.lib" "gdi32.lib" "winspool.lib" "comdlg32.lib" "advapi32.lib" "shell32.lib" "ole32.lib" "oleaut32.lib" "uuid.lib" "odbc32.lib" "odbccp32.lib" /MACHINE:X64 /SUBSYSTEM:CONSOLE /ERRORREPORT:none /NOLOGO /TLBID:1 /WX /LTCG /OPT:REF /INCREMENTAL:NO /OPT:ICF&&nvcc -O3 -Xptxas -v,-O3 -use_fast_math -D"_NVCC" -D"NDEBUG" -D"_WINDOWS" -D"_UNICODE" -D"UNICODE" -o gpu_kernel.dll --shared ../src/gpu/gpu_lstm_rnn.cu&&goto run
	goto end
)
:dbg
cl /c /permissive- /GS /W3 /Zc:wchar_t /Gm- /sdl /Zc:inline /fp:precise /D "_DEBUG" /D "_WINDOWS" /D "_UNICODE" /D "UNICODE" /errorReport:none /WX /Zc:forScope /Gd /Oi /FC /D /EHsc /nologo /diagnostics:column /ZI /Od /RTC1 /MDd ../src/main.c ../src/lstm/*.c&&link *.obj /OUT:lstm.exe /DYNAMICBASE "kernel32.lib" "user32.lib" "gdi32.lib" "winspool.lib" "comdlg32.lib" "advapi32.lib" "shell32.lib" "ole32.lib" "oleaut32.lib" "uuid.lib" "odbc32.lib" "odbccp32.lib" /MACHINE:X64 /SUBSYSTEM:CONSOLE /ERRORREPORT:none /NOLOGO /TLBID:1 /WX /DEBUG /INCREMENTAL&&nvcc -G -use_fast_math -D"_NVCC" -D"_DEBUG" -D"_WINDOWS" -D"_UNICODE" -D"UNICODE" -o gpu_kernel.dll --shared ../src/gpu/gpu_lstm_rnn.cu&&goto run
goto end
:run
del *.obj
del *.pdb
del *.exp
del *.ilk
del *.idb
del *.lib
cls
lstm.exe
:end
cd ../
set INCLUDE=%_INCLUDE%
