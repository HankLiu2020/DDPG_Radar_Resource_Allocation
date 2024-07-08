import numpy as np
import matplotlib.pyplot as plt
import math
import gym
from gym import spaces
import  torch

#定义雷达参数
env_radar_num=int(input("输入雷达数量："))
status_num=3#radar status 0 1 2

H=1000
#按照雷达数量，关于原点均布 
dis=5000
pos=[]#雷达阵地纵坐标
if env_radar_num%2==0:#偶数
    for i in range(env_radar_num):
        pos.append(dis*(i-(env_radar_num)//2+0.5))
else:
    for i in range(env_radar_num):    
        pos.append(dis*(i-(env_radar_num//2)))


#定义飞机参数

v=-300
speed=[v,0,0]
Pmin=10.0
Pmax=1000.0

jammer_mode_num=4#0,1,2,3
L=4.5E5 #飞机起始位置
pos_plane=[L,0,H]
total_steps=int(L/-v)

action_space=spaces.Box(low=np.array([0,Pmin,0]),high=np.array([2,Pmax,3]) )#连续动作
S_radars=[status_num]*env_radar_num#[3,3,3,3,3,...]
observe_space=spaces.MultiDiscrete([total_steps]+S_radars)

segment=[400E3,300E3,200E3,100E3]#单位m#修改了Rd1

#segment是分段函数的节点,从大到小
new_segment=[0.0]
#for tmp in segment[::-1]:#逆序，从小到大
segment.sort()
for tmp in segment:#从小到大
    new_segment.append(tmp)
new_segment.append(L)
#0 100 200 300 400 450


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
    def __init__(self,pos,speed,min,max):
        self.position = pos
        self.speed = speed

        self.L=pos[0]#针对本文，L就是起点x坐标 4.5E5 在reward3中用到
        self.v=speed[0]

        self.Pmin=min
        self.Pmax=max
        JetPlane.Plane_num+=1

        self.action_record=[]

    def get_position(self):
        return self.position

    def set_position(self,pos):
        self.position=pos

    def get_speed(self):
        return self.speed



    def record_action(self,action):
        #对多个雷达分别做不同动作，分别记录
        #干扰状态mode 0 1 2 3
        #action的结构：
        # [radar_index ,P连续 ,mode ], #对0雷达index 的 1干扰功率 和 2干扰模式
       

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
        
        P_now=self.action_record[-1][1]#最近一次，对于index雷达的干扰功率
        #功率reward2，pnow越大奖励越差
        return -1+(Pmax-P_now)/(Pmax-Pmin) #惩罚功率#修改了

    def calc_R3(self,new_segment):#index为0,1,2 segement为分段函数分段点
        Rd=self.position[0]#当前位置
        mode=self.action_record[-1][2]#最近一次，对于index雷达的干扰模式
        n_seg=len(new_segment)
      
        #r5近距离，单独写
        if Rd<=new_segment[1]:#100km内，mode3
            if mode==3:#只有密集假目标是+0.5
                return +0.5
            else:
                return -0.5
        #注意，以下部分修正了逻辑问题，后续需要merge到DQN
        for i in range(1,n_seg):#开始试分段,nseg=5
            #i=     1           2           3           4     
            #range  100-200     200-300     300-400     400-450
            #mode   3           2           1           0
            if new_segment[i]<Rd and new_segment[i+1]>=Rd: #从r4开始
                #位置分段判断判断
                if mode==(4-i):
                    #nt=(self.L-Rd)/(-self.v)#v=-300##运算复杂，需修改
                    return 5+((L-Rd))/(new_segment[i+1]-new_segment[i])#已改为：step越多，对应mode的reward越高
                    #return 0.5+nt*0.1*(5-i)#鼓励突防
                    #直接return，跳出循环探索
                else:
                    pass#继续遍历模式
        #for循环结束，说明没有找到对应分段
        return -0.5#-5

    #计算reward是飞机的方法！不需要指定雷达与序号
    def calc_reward(self,radar_list,new_segment):#计算本飞机对传入的雷达总共的reward R(S,A)其中A包含在action_record中
        radar_index=int(self.action_record[-1][0])#当前干扰的雷达号
        R:Radar=radar_list[radar_index]
        reward=self.calc_R1(R)+self.calc_R2()+self.calc_R3(new_segment)
        return reward

    #def calc_total_reward(self,radar_list,Rd_segment):
    #    reward=0
    #    for  i in range(len(radar_list)):
    #        reward+=self.calc_reward(radar_list,i,Rd_segment)
    #    return (reward)

    def calc_energy_ratio(self):
        #注意，对能量比例进行了修改 记得对DQN进行merge
        record_list=self.action_record#每一条都是[index,P,mode]
        Power=[]
        for record in record_list:
            Power.append(record[1].detach().cpu())

        #Pj_max=np.max(Power)
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
        max_energy=len(record_list)*Pmax

        return total_energy/(max_energy+0.001)

        
    def calc_Rmax(self,radar_list,action):#self:jetplane
        #计算最大探测距离 是 Pj干扰机功率 和 Rj 雷达-干扰机距离 的函数
        R:Radar
        R_max=[]
        for R in radar_list:
            Rj=distance(self.position,R.position)#雷达-干扰机距离
            jammed_radar_index=action[0]
            Pj=action[1]
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

    jetplane=JetPlane(pos_plane,speed,Pmin,Pmax)

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


#避免重复运算
hi=[env_radar_num-1,Pmax,jammer_mode_num-1]#修正了拉伸，Pmax-Pmin没必要
low=[0,Pmin,0]
span=[a-b+1 for a, b in zip(hi, low)]

def tanh2action(actionT):#把-1-1还原到action #弃用
    action=actionT.cpu().detach().numpy()
    #new_action=np.zeros(action.shape[0],dtype=np.int32)

    new_action=[np.floor(((a+1)/2)*(b)+c)  for a, b,c in zip(action,span, low)]
    new_action=np.array(new_action,dtype=np.int32)
    #for i in range(3):
    #    new_action[i]=np.floor(((action[i]+1)/2)*(span[i])+low[i])
        
    return new_action
    #action_tmp=torch.tensor(,device=device)*torch.tensor(action,device=device)+torch.tensor(,device=device)
    #return np.array(torch.floor(action_tmp).cpu(),dtype=int)

def multihead_action(action): #弃用
    
    P=np.floor(((action[1]+1)/2)*(span[1])+low[1])
    return [action[0],P,action[2]]

#def action2softmax(action,device):#把action压平到0-1
    #return (action-torch.tensor([0,Pmin,0],device=device))/torch.tensor([3,Pmax-Pmin,3],device=device)

def onehot_from_logits(logits, eps):
    ''' 生成最优动作的独热（one-hot）形式
        附带epsilon-greedy
        返回[[0., 1., 0.], [1., 0., 0.].....]'''
    argmax_acs = (logits == logits.max(dim=-1, keepdim=True)[0]).float()#1
    # 生成随机动作,转换成独热形式
    #eps的随机我放在takeaction函数里面了 这个forward需要处理批量输入
    '''rand_acs = torch.autograd.Variable(torch.eye(logits.shape[-1])[[
        np.random.choice(range(logits.shape[-1]), size=logits.shape[0])
    ]],
                                       requires_grad=False).to(logits.device)
    # 通过epsilon-贪婪算法来选择用哪个动作'''
    return argmax_acs
    '''torch.stack([
        argmax_acs[i] #if r > eps else rand_acs[i]
        for i, r in enumerate(torch.rand(logits.shape[0]))#i:顺序序号 r:随机数
    ])'''
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """从Gumbel(0,1)分布中采样"""#eps是计算偏移量Gi用的参数
    U = torch.autograd.Variable(tens_type(*shape).uniform_(),requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    """ 从Gumbel-Softmax分布中采样"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(
        logits.device)
    return torch.softmax(y / temperature, dim=-1)#1

def gumbel_softmax(logits, temperature,eps):
    """从Gumbel-Softmax分布中采样,并进行离散化"""
    y = gumbel_softmax_sample(logits, temperature)#给logits重新采样
    y_hard = onehot_from_logits(y,eps)#对重分布后的logits独热
    y = (y_hard.to(logits.device) - y).detach() + y
    # 返回一个y_hard的独热量,但是它的梯度是y,我们既能够得到一个与环境交互的离散动作,又可以
    # 正确地反传梯度
    return y

def step(radar_list:list,jetplane:JetPlane,action):
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

