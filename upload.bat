@echo off
chcp 65001 >nul
title GitHub 一键上传脚本
color 0A

echo ========================================================
echo        ComfyUI 插件自动上传工具 (GitHub)
echo ========================================================
echo.

echo [1/4] 正在检查当前状态...
git status
echo.
echo --------------------------------------------------------

:: 获取用户输入的提交信息
set /p msg="请输入本次更新说明 (直接回车默认为 Update): "
if "%msg%"=="" set msg=Update

echo.
echo [2/4] 正在添加文件 (git add)...
git add .

echo.
echo [3/4] 正在提交更改 (git commit)...
git commit -m "%msg%"

echo.
echo [4/4] 正在推送到 GitHub (git push)...
git push

echo.
echo ========================================================
if %errorlevel% == 0 (
    echo    恭喜！上传成功！ ^_^
) else (
    echo    上传失败，请检查网络或报错信息。 T_T
)
echo ========================================================
echo.
pause