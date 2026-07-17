# Linux, Module, Git, Python & PBS Command Cheat Sheet

## Linux Navigation Commands

| Command | Description |
|:--|:--|
| `pwd` | Print current working directory |
| `ls` | List files and directories |
| `ls -l` | Detailed file listing |
| `ls -a` | Show hidden files |
| `cd` | Change directory |
| `cd ..` | Go to parent directory |
| `cd ~` | Go to home directory |
| `tree` | Display directory structure (if installed) |

## Linux File Commands

| Command | Description |
|:--|:--|
| `mkdir` | Create a directory |
| `touch` | Create an empty file |
| `cp` | Copy files/directories |
| `mv` | Move or rename files |
| `rm` | Delete a file |
| `rm -r` | Delete a directory recursively |
| `cat` | Display file contents |
| `less` | View file page by page |
| `head` | Show first 10 lines |
| `tail` | Show last 10 lines |
| `tail -f` | Monitor a file in real time |
| `nano` | Edit a file |
| `find` | Search for files |
| `grep` | Search text inside files |

## Linux System Commands

| Command | Description |
|:--|:--|
| `whoami` | Display current username |
| `hostname` | Display system hostname |
| `date` | Display current date and time |
| `df -h` | Show disk usage |
| `du -sh` | Show directory size |
| `free -h` | Show memory usage |
| `top` | Monitor running processes |
| `ps -ef` | List running processes |
| `kill` | Terminate a process |

## Module Commands

| Command | Description |
|:--|:--|
| `module avail` | List all available software modules |
| `module list` | Show currently loaded modules |
| `module load <module>` | Load a software module |
| `module unload <module>` | Unload a software module |
| `module purge` | Unload all loaded modules |
| `module use <path>` | Add a custom module search path |
| `module spider <module>` | Search for available versions (if supported) |
| `module help <module>` | Display help for a module |

## Python & Virtual Environment Commands

| Command | Description |
|:--|:--|
| `python3 --version` | Display Python version |
| `python3 -m pip --version` | Display pip version |
| `python3 -m venv venv` | Create a virtual environment |
| `source venv/bin/activate` | Activate virtual environment |
| `deactivate` | Exit virtual environment |
| `pip3 install -r requirements.txt` | Install project dependencies |
| `pip3 list` | List installed packages |
| `pip3 freeze` | Export installed packages |

## Git Commands

| Command | Description |
|:--|:--|
| `git clone` | Clone a repository |
| `git status` | Show repository status |
| `git pull` | Download latest changes |
| `git log` | Show commit history |

## PBS Client Commands

| Command | Description |
|:--|:--|
| `qsub` | Submit batch jobs |
| `qstat` | View queues and jobs |
| `qstat -u $USER` | View your jobs |
| `qdel` | Delete/cancel batch jobs |
| `qhold` | Hold batch jobs |
| `qrls` | Release job holds |
| `qalter` | Modify queued batch jobs |
| `qrerun` | Rerun a batch job |
| `qmove` | Move batch jobs |
| `qrun` | Start a queued batch job |
| `tracejob` | Trace job execution |
| `momctl` | Manage/diagnose MOM daemon |
| `pbsdsh` | Launch tasks within a parallel job |
| `pbsnodes` | View/modify compute node status |
| `qchkpt` | Checkpoint batch jobs |
| `qgpumode` | Specify GPU mode |
| `qgpureset` | Reset GPU |
| `qmgr` | Manage PBS configuration |
| `qsig` | Send signal to a batch job |
| `qterm` | Shutdown PBS server daemon |

## Useful PBS Output Commands

| Command | Description |
|:--|:--|
| `cat job.o<JobID>` | View standard output |
| `cat job.e<JobID>` | View standard error |
| `less job.o<JobID>` | View output interactively |
| `tail -f job.o<JobID>` | Monitor output live |

