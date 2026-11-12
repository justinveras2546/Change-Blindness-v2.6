@echo off
rem Build a local binary (no SDL install needed: it is embedded).
cd /d "%~dp0"
set CGO_ENABLED=0
go build -trimpath -ldflags="-s -w" -o Change-Blindness.exe .
echo built Change-Blindness.exe - run it with: Change-Blindness.exe -s 1
