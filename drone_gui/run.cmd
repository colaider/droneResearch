@echo off
REM Start the control panel on Windows. The default (SSH) transport needs no ROS and no WSL -
REM just Python 3 and the ssh client Windows ships with.
REM
REM   run.cmd            http://localhost:8080
REM   run.cmd --open     ...and open a browser
REM   run.cmd --login    first run here: make an SSH key and install it on the board
cd /d "%~dp0"
python -m drone_gui %*
