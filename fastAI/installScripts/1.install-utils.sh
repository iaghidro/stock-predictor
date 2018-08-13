# set LC_ALL in /etc/default/locale
cp /etc/default/locale .
echo "LC_ALL=\"en_US.UTF-8\"" >> locale
sudo cp locale /etc/default/locale
rm locale

# insert here the installation of any optional utility libraries
sudo apt --assume-yes install tmux
sudo apt --assume-yes install unzip
sudo apt --assume-yes install ffmpeg