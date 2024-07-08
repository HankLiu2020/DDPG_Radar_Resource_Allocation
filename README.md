# 基于强化学习的雷达干扰资源分配
&emsp;本项目对干扰机-雷达一对多场景下的干扰突防问题进行了强化学习建模，并采用DQN与DDPG两种算法，对雷达干扰资源分配问题进行了仿真，并对仿真结果进行了分析。

## 项目目录结构介绍
基于强化学习的雷达干扰资源分配  
   |—__DQN__&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;// 使用DQN算法实现干扰资源分配  
   |——RadarEnv.py&emsp;&emsp;&emsp;&emsp;&emsp;//针对DQN进行动作离散化的雷达-干扰机交互环境  
   |——rl_utils.py&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;// 强化学习常用函数  
   |——run_DQN.ipynb&emsp;&emsp;&emsp;&emsp;//基于DQN的突防仿真实现  
   |—__DDPG__&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;// 使用DDPG算法实现干扰资源分配  
   |——RadarEnv_ddpg.py &emsp;&emsp;//采用混合动作空间的雷达-干扰机交互环境  
   |——rl_utils.py&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;// 强化学习常用函数  
   |——tryDDPG.ipynb&emsp;&emsp;&emsp;&emsp;//基于DDPG的突防仿真实现  
   |—__Patent__&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;//专利相关的仿真
   |——RadarEnv_ddpg_n.py &emsp;&emsp;//采用混合动作空间的多雷达-干扰机交互环境  
   |——rl_utils.py&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;// 强化学习常用函数  
   |——tryDDPG_n.ipynb&emsp;&emsp;&emsp;&emsp;//基于DDPG的多雷达-单干扰机突防仿真实现  
   |—__requirements.txt__&emsp;&emsp;&emsp;&emsp;&emsp;//  依赖包  
   |—__README.md__&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;// 项目介绍  


## 开发工具环境搭建

&emsp;Python 3.8 + Pytorch 2.1
&emsp;环境使用的依赖包详见requirements.txt

## 项目运行方式
&emsp;使用Jupyter Notebook在配置好的虚拟环境中运行run_DQN.ipynb与tryDDPG.ipynb文件，运行结果将以内嵌形式展示。

## 项目开发人员
&emsp;刘骥远[hank2016@qq.com]

## 指导老师
&emsp;王博[wangbo@hdu.edu.cn]
