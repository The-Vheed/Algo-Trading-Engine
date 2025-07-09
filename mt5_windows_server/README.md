# trading-bot

### Download and install mt5
``` shell
curl -O https://download.mql5.com/cdn/web/22698/mt5/derivsvg5setup.exe
./derivsvg5setup.exe
```

### Download and install git
``` shell
curl -O https://github.com/git-for-windows/git/releases/download/v2.44.0.windows.1/Git-2.44.0-64-bit.exe
./Git-2.44.0-64-bit.exe
```
### Download and install python
``` shell
curl -O https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
./python-3.10.11-amd64.exe
```
### Open port 80 on firewall
``` shell
netsh advfirewall firewall add rule name="Open Port 80" dir=in action=allow protocol=TCP localport=80
```
### Clone git repo and install dependencies
``` shell
git clone https://$GithubPAT@github.com/The-Vheed/tb-windows.git
cd tb-windows
py -m pip install -r windows_requirements.txt
py windows_server.py
```