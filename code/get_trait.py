import numpy as np
from scipy.fftpack import fft
from scipy.stats import skew 


#把成分空间和模式空间提取出特征空间
def get_trait(components_space,pattern_space,freq=200):

    components_trait = np.zeros((26,components_space.shape[1]))
    #时间序列
    #最大值
    components_trait[0,:] = max_amplitude = components_space.max(axis=0)

    #幅度范围
    components_trait[1,:] = range_amplitude = components_space.max(axis=0) - components_space.min(axis=0)

    #最大一阶导数
    First_Derivative = np.zeros((components_space.shape))
    for i in range(len(components_space)-1):
        First_Derivative[i,:] = components_space[i+1,:] - components_space[i,:]
    components_trait[2,:] = max_First_Derivative = First_Derivative.max(axis=0)

    #香农熵
    p_matrix = np.ones((15,components_space.shape[1]))
    digitized = np.ones((components_space.shape))
    for i in range(components_space.shape[1]):
        bins = np.linspace(components_space[:,i].min(), components_space[:,i].max()+0.00000000000001,16) #15个bins
        digitized[:,i] = np.digitize(components_space[:,i], bins)
       
    for i in range(15):
        bool_matrix=digitized==i+1
        p_matrix[i,:] = bool_matrix.sum(axis=0)/len(bool_matrix)
    components_trait[3,:] = shannon_entropy = -np.sum(p_matrix * np.log2(p_matrix + 0.00000000000001),axis=0)

    #计算局部方差，均值，偏度,0.5s
    local_variance = np.zeros((100,components_space.shape[1]))
    local_mean = np.zeros((100,components_space.shape[1]))
    local_skewness = np.zeros((100,components_space.shape[1]))
    local_index = -1
    for i in range(int(len(components_space) / (freq/2))):
        local_index = local_index + 1
        old_i = int(i)
        i = int(i+(freq/2))

        if i > len(components_space):
            i = len(components_space) 
        local = components_space[old_i:i,:]

        local_variance[local_index,:] = np.var(local,axis=0)
        local_mean[local_index,:] = np.mean(local,axis=0)
        local_skewness[local_index,:] = skew(local,bias=False)

    #去除全为零的行   
    local_variance = local_variance[[not np.all(local_variance[i] == 0) for i in range(local_variance.shape[0])],:]
    local_mean = local_mean[[not np.all(local_mean[i] == 0) for i in range(local_mean.shape[0])],:]
    local_skewness = local_skewness[[not np.all(local_skewness[i] == 0) for i in range(local_skewness.shape[0])],:]
    #计算局部方差，均值，偏度的平均值和方差
    local_trait = np.zeros((6,components_space.shape[1]))
    local_trait[0,:] = vlv = np.var(local_variance,axis=0)
    local_trait[1,:] = vlm = np.var(local_mean,axis=0)
    local_trait[2,:] = vls = np.var(local_skewness,axis=0)
    local_trait[3,:] = mlv = np.mean(local_variance,axis=0)
    local_trait[4,:] = mlm = np.mean(local_mean,axis=0)
    local_trait[5,:] = mls = np.mean(local_skewness,axis=0)

    components_trait[4:10,:] = local_trait


    #IC能量
    #self.IC_energy = (abs(self.components**2)).sum(axis=0)  

    #峰度
    E2 = (components_space**2/len(components_space)).sum(axis=0)
    E4 = (components_space**4/len(components_space)).sum(axis=0)
    components_trait[10,:] = Kurtosis = E4-3*(E2**2)     

    #频谱
    PSD = np.zeros((components_space.shape))
    i_freq = freq/len(PSD)
    for i in range(components_space.shape[1]):
        x = components_space[:,i]
        PSD[:,i] = abs(fft(x)/len(PSD)*2)
    PSD_1_3 = []
    PSD_4_7 = []
    PSD_8_13 = []
    PSD_14_30 = []
    PSD_31_45 = []
    #提取特定频段做分析
    PSD_1_3 = np.mean(PSD[int(1*(1/i_freq)):int(3*(1/i_freq)),:] ,axis=0)
    PSD_4_7 = np.mean(PSD[int(4*(1/i_freq)):int(7*(1/i_freq)),:] ,axis=0)
    PSD_8_13 = np.mean(PSD[int(8*(1/i_freq)):int(13*(1/i_freq)),:] ,axis=0)
    PSD_14_30 = np.mean(PSD[int(14*(1/i_freq)):int(30*(1/i_freq)),:] ,axis=0)
    PSD_31_45 = np.mean(PSD[int(31*(1/i_freq)):int(45*(1/i_freq)),:] ,axis=0)
    log_PSD = np.zeros((5,PSD.shape[1]))
    log_PSD[0,:] = np.log(PSD_1_3)
    log_PSD[1,:] = np.log(PSD_4_7)
    log_PSD[2,:] = np.log(PSD_8_13)
    log_PSD[3,:] = np.log(PSD_14_30)
    log_PSD[4,:] = np.log(PSD_31_45)
    components_trait[11:16,:] = log_PSD

    #频谱上特殊的点
    #3hz
    log_PSD_3 = np.log(PSD[int(3*(1/i_freq)),:])
    components_trait[16,:] = log_PSD_3

    #PSD_norm = []
    #PSD_norm = PSD_1_60/PSD_1_60.max(axis=0) 
    #提取特定频段做眼电信号分析
    #self.PSD_norm_1 = PSD_norm[int((l_freq-0.5)*(1/i_freq)):int((h_freq-0.5)*(1/i_freq)),:]      #  samples x components

    #模式
    #模式的范围
    components_trait[17,:] = range_pattern = (np.max(pattern_space,axis=0) - np.min(pattern_space,axis=0))

    #最大与最小电极的欧氏距离
    position_pattern = np.zeros((1,pattern_space.shape[1]))

    position_extrme = np.zeros((2,pattern_space.shape[1]))
    position_extrme[0,:] = np.argmax(pattern_space, axis=0)
    position_extrme[1,:] = np.argmin(pattern_space, axis=0)

    for i in range(pattern_space.shape[1]):
        if ((position_extrme[0,i]==0 or position_extrme[1,i]==0) and (position_extrme[0,i]==1 or position_extrme[1,i]==1)) or ((position_extrme[0,i]==2 or position_extrme[1,i]==2) and (position_extrme[0,i]==3 or position_extrme[1,i]==3)):
            position_pattern[0,i] = 1
        elif ((position_extrme[0,i]==1 or position_extrme[1,i]==1) and (position_extrme[0,i]==2 or position_extrme[1,i]==2)):
            position_pattern[0,i] = 2
        elif ((position_extrme[0,i]==0 or position_extrme[1,i]==0) and (position_extrme[0,i]==2 or position_extrme[1,i]==2)) or ((position_extrme[0,i]==1 or position_extrme[1,i]==1) and (position_extrme[0,i]==3 or position_extrme[1,i]==3)):
            position_pattern[0,i] = 3
        else:
            position_pattern[0,i] = 4
    components_trait[18,:] = position_pattern

    #最大模式
    components_trait[19,:] = max_pattern = np.max(pattern_space,axis=0)

    #减去平均值后的模式
    components_trait[20:24,:] = c_pattern_space = pattern_space - np.mean(pattern_space,axis=0)

    #增加的特征
    # #10hz
    log_PSD_10 = np.log(PSD[int(10*(1/i_freq)),:])
    components_trait[24,:] = log_PSD_10
    
    #方差
    A = pattern_space.std(axis=0)
    X = np.zeros((components_space.shape))
    for iii in range(components_space.shape[1]):
        X[:,iii] = A[iii] * components_space[:,iii]
    components_trait[25,:] = X.var(axis=0)


    print('获得特征空间')
    return components_trait
'''
def train(components_trait):
    np.random.shuffle(components_trait)

    #train_data = components_trait[:236,:]
    #test_data = components_trait[236:,:]
    #train_label = components_label[:236]
    #test_label = components_label[236:]
    svm_model = SVC(kernel = 'rbf')
    print(cross_val_score(svm_model,components_trait[:,1:],components_trait[:,0],cv=10,n_jobs=6))
    svm_model.fit(train_data,train_label)

    pre_train = svm_model.predict(train_data)
    accuracy_train = accuracy_score(train_label,pre_train)
    print('训练精度：')
    print(accuracy_train)

    pre_test = svm_model.predict(test_data)
    accuracy_test = accuracy_score(test_label,pre_test)
    print('测试精度：')
    print(accuracy_test)
    return svm_model
'''
#获取特征
f_components_space = r'../data1/components_space_5.csv'
with open(f_components_space,encoding = 'utf-8') as pp:
    components_space = np.loadtxt(pp,str,delimiter = ",",comments='%')
components_space=np.array(components_space,dtype='float64')

f_pattern_space = r'../data1/pattern_space_5.csv'
with open(f_pattern_space,encoding = 'utf-8') as pp:
    pattern_space = np.loadtxt(pp,str,delimiter = ",",comments='%')
pattern_space=np.array(pattern_space,dtype='float64')

trait_space = get_trait(components_space,pattern_space)
np.savetxt('../data1/trait/trait_space_5.csv', trait_space, delimiter=',')
print(components_space.dtype)


#训练
'''f_trait_space = r'../data1/trait/trait_space.csv'
with open(f_trait_space,encoding = 'utf-8') as pp:
    trait_space = np.loadtxt(pp,str,delimiter = ",",comments='%')
trait_space=np.array(trait_space,dtype='float64')

f_components_label = r'../data/components_label_1+2.csv'
with open(f_components_label,encoding = 'utf-8') as pp:
    components_label = np.loadtxt(pp,str,delimiter = ",",comments='%')
components_label = np.array(components_label,dtype='int32')

model = train(trait_space.T,)'''