# Dual Boot Win 10 and Ubuntu 16.04

## Create Win10 Installation USB
Refer to [This Post](https://answers.microsoft.com/en-us/windows/forum/windows_10-windows_install/clean-install-windows-10/1c426bdf-79b1-4d42-be93-17378d93e587) <br/>
在[这里](https://www.microsoft.com/en-us/software-download/windows10ISO)下载media creation tool

# 安装Win10
1. 开机狂按F2进入BIOS设置
2. `Boot`>`UEFI Mode`选enable，保存选项并退出
3. 开机狂按F7选择win10安装盘启动
4. 根据提示一路“下一步”进入目标安装磁盘选择
5. 将目标盘中所有分区删除（这里我选择的磁盘0即SSD作为目标盘），这时磁盘0应该被合并成未被分配的磁盘0，选择该磁盘然后一路按照提示安装

# WIN10设置
请参考[这个帖子](http://www.everydaylinuxuser.com/2015/11/how-to-install-ubuntu-linux-alongside.html)
1. 右键点击Win菜单，选择`电源选项`>`其它电源选项`>"关闭快速启动"
2. 右键点击Win菜单，选择`磁盘管理`，右键点击C盘所在分区，选择"压缩卷"，在弹出的窗口中分配Ubuntu所需空间大小（无法分配超过C盘空间的大小，所以我直接选择了最大的值，大约118G）

# 用[rufus](https://rufus.akeo.ie/)制作Ubuntu安装盘
参考[官方教程](https://tutorials.ubuntu.com/tutorial/tutorial-create-a-usb-stick-on-windows)
 
# 安装Ubuntu 16.04
参考[这个帖子](https://blog.csdn.net/qq_39105012/article/details/80427792)
1. **注意：最坑的来了**开机狂按F2进入BIOS设置，一次选择`Advanced`>`Advanced Chipset Control`>`MSHybrid or DISCRETE Switch`选择`DISCRETE`。保存设置并重启
2. 狂按F7选择Ubuntu安装盘
3. 选择"Try Ubuntu without installation"进入图形界面，如果不能进入图形界面可以按[以下步骤](https://askubuntu.com/questions/760934/graphics-issues-after-while-installing-ubuntu-16-04-16-10-with-nvidia-graphics)操作
    - 重启进入那个有"Try Ubuntu without installation"的界面(GRUB).
    - 按“e”键，然后找到开头是linux的那行并把光标移动到最后（这行有可能被断成两行显示）.
    - 输入`nouveau.modeset=0`.
    - 按 `F10`或`Ctrl+X`应该就可以进入Ubuntu图形界面了
4. 双击桌面上的"Install Ubuntu 16.04"的图标按提示进行安装

# Ubuntu安装完毕后其他事项
## 更新
`$ sudo apt update`<br/>
`$ sudo apt upgrade`<br/>
如果遇到了这样的问题
```
E: Could not get lock /var/lib/dpkg/lock - open (11 Resource temporarily unavailable)
E: Unable to lock the administration directory (/var/lib/dpkg/) is another process using it?  
```
可以尝试用`$ sudo rm /var/lib/apt/lists/* -vf`来解决
## 安装nvidia显卡驱动
`$ sudo add-apt-repository ppa:graphics-drivers/ppa`<br/>
`$ sudo apt update`<br/>
`$ sudo apt install nvidia-396`<br/>
