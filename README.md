# bomberman_rl
Setup for a project/competition amongst students to train a winning Reinforcement Learning agent for the classic game Bomberman.


For google colab:
```
! rm -Rdf /root/.ssh/*
! mkdir /root/.ssh

! tar -xvzf ssh.tar.gz

! cp id_rsa /root/.ssh && cp id_rsa.pub /root/.ssh && rm -rf id_rsa* && rm -rf ssh.tar.gz ! chmod 700 /root/.ssh

! ssh-keyscan github.com >> /root/.ssh/known_hosts
! chmod 644 /root/.ssh/known_hosts

! git config --global user.email "email"
! git config --global user.name "name"

! ssh -vT git@github.com

! git clone git@github.com:Akatuoro/bomberchamp.git
```
