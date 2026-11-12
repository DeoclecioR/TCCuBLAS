@echo off
REM ---------------------------------------------------------------------------
REM Compila o projeto no Windows: localiza o MSVC (Visual Studio / Build Tools),
REM configura o ambiente x64 e chama o mingw32-make.
REM
REM O CUDA 12.x exige um toolset do MSVC compativel (v143, VS 2022, 14.3x/14.4x).
REM Toolsets mais novos (ex.: 14.5x do Visual Studio 2026) derrubam o nvcc.
REM O script usa o v143 automaticamente quando ele estiver instalado.
REM
REM Uso:  compilar_windows.bat            (compila para sm_61)
REM       compilar_windows.bat CUDA_ARCH=sm_75
REM       compilar_windows.bat clean
REM ---------------------------------------------------------------------------
setlocal

set "INSTALADOR=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer"
if not exist "%INSTALADOR%\vswhere.exe" (
    echo [erro] Visual Studio / Build Tools nao encontrado.
    echo Instale o Build Tools com a carga "Desenvolvimento para desktop com C++".
    exit /b 1
)

REM Caminho curto (8.3) do vswhere: evita problemas com espacos e parenteses.
for %%I in ("%INSTALADOR%\vswhere.exe") do set "VSW=%%~sI"
for /f "usebackq delims=" %%i in (`%VSW% -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"

if not defined VSDIR (
    echo [erro] Componente C++ do MSVC ausente. Instale a carga "Desenvolvimento para desktop com C++".
    exit /b 1
)

REM Prefere o toolset v143 (14.3x / 14.4x) se instalado.
set "VCARGS="
for /d %%d in ("%VSDIR%\VC\Tools\MSVC\14.4*" "%VSDIR%\VC\Tools\MSVC\14.3*") do (
    if not defined VCARGS set "VCARGS=-vcvars_ver=%%~nxd"
)
if defined VCARGS (
    echo [info] usando toolset MSVC %VCARGS%
) else (
    echo [aviso] toolset v143 nao instalado; usando o mais recente, que pode nao ser aceito pelo nvcc.
)

call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" %VCARGS% >nul || exit /b 1

where nvcc >nul 2>nul || (echo [erro] nvcc nao encontrado no PATH. Instale o CUDA Toolkit 12.x. & exit /b 1)

mingw32-make %*
exit /b %errorlevel%
