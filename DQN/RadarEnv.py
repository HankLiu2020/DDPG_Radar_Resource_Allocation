import numpy as np
import matplotlib.pyplot as plt
import math
import gym
from gym import spaces

#定义雷达参数
env_radar_num=3#radar num
status_num=3#radar status 0 1 2

H=1000
pos=[5000,0,-5000]#雷达阵地纵坐标


#定义飞机参数

v=-300
speed=[v,0,0]
Pmin=10
Pmax=1000
Pn=11#雷达最小最大功率，功率划分n份
jammer_mode_num=4#0,1,2,3
L=4.5E5 #飞机起始位置
pos_plane=[L,0,H]
total_steps=int(L/-v)

action_space=spaces.MultiDiscrete([env_radar_num, Pn , jammer_mode_num])#start=
S_radars=[status_num]*status_num#一个tmp 凑一个数组
observe_space=spaces.MultiDiscrete([total_steps]+S_radars)

segment=[400E3,300E3,200E3,100E3]#单位m#修改了Rd1

#segment是分段函数的节点,从大到小
new_segment=[0.0]
#for tmp in segment[::-1]:#逆序，从小到大
segment.sort()
for tmp in segment:#从小到大
    new_segment.append(tmp)
new_segment.append(L)



def distance(pos1,pos2):#三维距离
    sum=0
    for i in range(len(pos1)):
        sum+=np.square(pos1[i]-pos2[i])
    return abs(np.sqrt(sum))

class Radar:
    is_over=False#制导后终止迭代
    radar_num=0 #全局变量 用于雷达计数

    def __init__(self,pos):
        Radar.is_over=False#初始化雷达状态,本轮刚开始
        self.index=Radar.radar_num
        Radar.radar_num+=1
        self.position = pos
        #雷达状态：status 0搜索 1跟踪 2制导
        self.status = 0
        #雷达在时间序列上是否检测到目标 0为未检测到
        self.target_detect=[0,0,0]#序号从3即为第一次
        self.status_record=[0,0]
        self.status_record.append(self.status)
        
    def detect_and_status(self, jetplane,R_max):
        if distance(self.position,jetplane.position)<R_max:
            #距离小于最大探测距离，即为发现目标
            self.target_detect.append(1)
        else:
            self.target_detect.append(0)

        if self.status == 0:#搜索
            #最近四次有三次发现目标
            if sum(self.target_detect[-4:]) >= 3:
                self.status = 1#跟踪
            else:
                pass #自环
        elif self.status==1:#跟踪
            #三次均未发现目标
            if sum(self.target_detect[-3:]) == 0:
                #回到搜索
                self.status = 0
            elif sum(self.target_detect[-3:]) >= 2:
                #制导！
                self.status = 2
                Radar.is_over = True
            else:
                pass#自环
        elif self.status==2:
            Radar.is_over=True#制导状态 结束
        
        self.status_record.append(self.status)

        if Radar.is_over==True: 
            return 1 #制导状态，本轮结束！
        else: return 0

    def get_position(self):
         return self.position
         
    def get_status(self):#获取雷达状态，而不是state
         return self.status

    def get_status_record(self):
         return self.status_record
    
    def __del__(self):
        Radar.radar_num -= 1


class JetPlane:
    Plane_num=0
    def __init__(self,pos,speed,min,max,Pn):
        self.position = pos
        self.speed = speed

        self.L=pos[0]#针对本文，L就是起点x坐标 4.5E5 在reward3中用到
        self.v=speed[0]

        self.Pmin=min
        self.Pmax=max
        JetPlane.Plane_num+=1

        self.action_record=[]

        #将功率平均分为n份，并生成序列
        self.P_seq=[]
        interval=(self.Pmax-self.Pmin)/(Pn)
        
        for i in range(Pn):
            self.P_seq.append(self.Pmin+interval*i)

    def get_position(self):
        return self.position

    def set_position(self,pos):
        self.position=pos

    def get_speed(self):
        return self.speed

    def get_P_seq(self):#返回离散功率
        return self.P_seq


    def record_action(self,action):
        #对多个雷达分别做不同动作，分别记录
        #干扰状态mode 0 1 2 3
        #action的结构：
        # [radar_index ,self.P_seq[P_index] ,mode ], #对0雷达index 的 1干扰功率 和 2干扰模式
       

        self.action_record.append(action)
        
    def get_action_record(self):
        return self.action_record

    #需要在当前步骤 更新完雷达状态 后 计算
    def calc_R1(self,R:Radar,):#计算每一个雷达的reward1
        
        status=R.get_status_record()[-2:]
        if status[-1]>status[-2]:#当前雷达状态增大
            return -5
        elif status[-1]<status[-2]:
            return 5
        else:
            return 1

    def calc_R2(self):#计算对index雷达的干扰P功率的reward2
        #radar_index=self.action_record[-1][0]
        P_now=self.action_record[-1][1]#最近一次，对于index雷达的干扰功率
        #P_now=self.P_seq[P_now]
        #功率reward，pnow越大奖励越差
        #return -0.5+0.1*((self.Pmax-P_now)/(self.Pmax-self.Pmin)*10//1)#分为10份,0.1步进
        return (Pn-1-P_now)*0.1 #修改了 不鼓励高功率

    def calc_R3(self,new_segment):#index为0,1,2 segement为分段函数分段点
        Rd=self.position[0]#当前位置
        mode=self.action_record[-1][2]#最近一次，对于index雷达的干扰模式
        n_seg=len(new_segment)
      
        #r5近距离，单独写
        if Rd<=new_segment[1]:
            if mode==3:#只有密集假目标是+0.5
                return +0.5
            else:
                return -0.5

        for i in range(1,n_seg):
            if new_segment[i]<Rd and new_segment[i+1]>=Rd: #从r4开始
                #模式判断
                if mode==(4-i):
                    nt=(self.L-Rd)/(-self.v)#v=-300
                    return 5+(self.L+(-self.v)*nt)/(new_segment[i+1]-new_segment[i])#已改为：step越多，对应mode的reward越高
                    #return 0.5+nt*0.1*(5-i)#鼓励突防
                else:
                    pass
        return -0.5#-5

    #计算reward是飞机的方法！不需要指定雷达与序号
    def calc_reward(self,radar_list,new_segment):#计算本飞机对传入的雷达总共的reward R(S,A)其中A包含在action_record中
        radar_index=self.action_record[-1][0]#当前干扰的雷达号
        R:Radar=radar_list[radar_index]
        reward=self.calc_R1(R)+self.calc_R2()+self.calc_R3(new_segment)
        return reward

    #def calc_total_reward(self,radar_list,Rd_segment):
    #    reward=0
    #    for  i in range(len(radar_list)):
    #        reward+=self.calc_reward(radar_list,i,Rd_segment)
    #    return (reward)

    def calc_energy_ratio(self):
        record_list=self.action_record#每一条都是[index,P,mode]
        Power=[]
        for record in record_list:
            Power.append(record[1])

        Pj_max=np.max(Power)
        """for i in range(4):#雷达mode一共4种
            Power.append([])

        for record in record_list:
            #一条record由当前步骤中 一个雷达的index P和mode组成
            #for action in record:
                P=record[1]
                mode=record[2]
                Power[mode].append(P)#桶排
        #计算每个mode下的总能量
        Pj_max=np.argmax(Power)
        Energy=[0,0,0,0]
        for i in range(4):
            Energy[i]=sum(Power[i])"""
        total_energy=sum(Power)
        max_energy=len(record_list)*Pn

        return total_energy/(max_energy+0.001)

        
    def calc_Rmax(self,radar_list,action):#self:jetplane
        #计算最大探测距离 是 Pj干扰机功率 和 Rj 雷达-干扰机距离 的函数
        R:Radar
        R_max=[]
        for R in radar_list:
            Rj=distance(self.position,R.position)#雷达-干扰机距离
            jammed_radar_index=action[0]
            Pj=Pmin+((Pmax-Pmin)/Pn)*action[1]
            if R.index==jammed_radar_index:#干扰本雷达
                
                
                R_max.append(math.pow((9.718E12*math.pow(Rj,2))/Pj,0.25))
        
            else:#不是正对该雷达
                #R_max.append(math.pow((9.718E12*math.pow(Rj,2))/0.5E4*Pj,0.25))
                R_max.append(200E3)
        
        return R_max


        

    
def Env_init():#初始化环境
    #定义雷达
    radar=[]
    Radar.radar_num=0
    
    for i in range(env_radar_num):
        radar.append(Radar([0, pos[i],H]))

    jetplane=JetPlane(pos_plane,speed,Pmin,Pmax,Pn)

    return radar,jetplane#返回雷达对象数组和飞机对象

#不在任何 类 内
def get_current_state(radar_list,plane:JetPlane):
    #S_t:[Rd距离雷达1（中间那个)的距离step， status1, 2, 3各个雷达工作状态 ]
    R:Radar 
    Rd:int
    status=[]
    for R in radar_list:
        status.append(R.get_status())
    Rd=pos2step(plane.position,speed,L)#当前位置-->step
    return [Rd]+status


def pos2step(pos,speed,L):
    Rd=pos[0]#x
    Vx=-speed[0]
    step:int
    step=(int)(L-Rd)/Vx#从0开始
    return step

def step2pos(step_now:int,speed,L):
    Vx=speed[0]#负值
    H=1000#暂时这么写 以后改
    pos=[L+Vx*step_now,0,H]
    return pos

def action2dis(action):
    #action=[index,P,mode]
    #action_size
    [a,b,c]=np.array(action_space.nvec)
    x,y,z=action
    return x+y*a+z*a*b

def dis2action(index):
    [a,b,c]=action_space.nvec
    dim2:int=index%(a*b)
    z:int=index//(a*b)
    x:int=dim2%a
    y:int=dim2//a
    return [x,y,z]

def step(radar_list:list,jetplane:JetPlane,Rd_segment,action):
    #先更新位置
    pos=jetplane.get_position()
    now_pos = [pos[i] + speed[i] for i in range(len(pos))]

    jetplane.set_position(now_pos)

    '''
    #对action进行解析，可能没用
    radar_index=action[0]   #干扰机锁定雷达index
    jam_power=action[1]     #干扰机功率
    jam_mode=action[2]      #干扰机模式
    '''
    #记录action
    jetplane.record_action(action)

    #采取action
    current_radar_status=[]
    R:Radar
    
    R_max_list=jetplane.calc_Rmax(radar_list,action)#计算最大可探测距离Rmax，用来更新状态
    #for index in range(len(radar_list)):

    is_over=0#初始化累加
    for R in (radar_list):
        #逐一更新雷达状态status
        is_over+=R.detect_and_status(jetplane,R_max_list[R.index])
        ''' #current_radar_status.append(R.get_status())#更新雷达状[status1,2,3]
    
    
    now_step=pos2step(now_pos,jetplane.speed,Rd_segment)
    next_state=[now_step]+(current_radar_status)
    '''
    #直接获取Rd步+雷达状态
    next_state=get_current_state(radar_list,jetplane)
    #返回status [Rd,s1,s2,s3]
    
    
    #计算reward
    current_reward=jetplane.calc_reward(radar_list,new_segment)

    if is_over==0 and jetplane.position[0]>0:#没被制导且未飞跃阵地
        done=False
    else:
        done=True
    
    return next_state,current_reward,done



#问题1：Rmax怎么重新定义 Rmax是否和 被干扰的雷达 有关 干不干扰
#问题2：action函数有点古怪 
# status这种状态空间 是不是要全部摊平 √已经摊平了
# 再给神经网络拟合
#问题3：神经网络做法

#双dqn也没写


'''7. 注意CPU和GPU之间频繁的数据传输
小心使用tensor.cpu()和tensor.cuda()频繁地将张量从GPU和CPU之间相互转换。对于.item()和.numpy()也是一样，用.detach()代替。

如果你正在创建一个新的张量，你也可以使用关键字参数device=torch.device('cuda:0')直接将它分配给你的GPU。

如果你确实需要传输数据，在传输后使用.to(non_blocking=True)可能会很有用，只要你没有任何同步点。

如果你真的需要，你可以试试Santosh Gupta的SpeedTorch，虽然不是很确定在什么情况下可以加速。'''

#1.13现在的状态#
'''
给这个论文分成两部分，环境+DQN
环境上Rmax存在比较大的问题
一次能干扰几个雷达
如果能干扰多个 动作空间爆炸了
如果只干扰一个 那不被干扰的是常数
如果一带多，那天线的角度增益又未知

只要能轮流干扰，就始终能突防 增加雷达数量会导致死限

环境和DQN交互的部分 以上的问题我已近自作主张做掉了
不知道给action压平这个做法对不对
1神经网络是不是太简单了
2按照标准的步骤写的 论文里面提及很少 可能有问题 还是需要一个可靠的Rmax
3运算速度 gpu没啥动静
'''


