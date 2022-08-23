import mne
from sklearn.decomposition import FastICA
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft


#输入数据获得成分空间和模式空间，通过观察手动制作标签
def get_raw_data(freq=200,num=20,t=5,n_com=4):
    file_path = r'C:\Users\123\EEG\sample\008心影冥想B2020_12_22_18_28_26_data.txt'
    with open(file_path,encoding = 'utf-8') as p:
        data = np.loadtxt(p,str,delimiter = ",",skiprows=25,comments='%')
    data = data[8:,1:5].T

    #注释信息
    info = mne.create_info(ch_names=['AF7', 'Fp1','Fp2','AF8'],
                           ch_types=['eeg'] * 4,
                           sfreq=freq)

    raw_array = mne.io.RawArray(data, info)
    raw_array.set_montage('standard_1020') # 定位电极
    raw_array.filter(l_freq=1,h_freq=60)   #1-60hz
    raw_array.notch_filter(freqs=50) #凹陷滤波50hz干扰

    pattern_space = np.zeros((4,n_com*num))
    raw_data_space = np.zeros((freq*t+1,4*num))
    components_space = np.zeros((freq*t+1,n_com*num))
    for i in range(num):
        start = np.random.randint(30,930)
        raw_crop = raw_array.copy()
        raw_crop.crop(tmin=start,tmax=start+t)
        raw_np_crop, times = raw_crop[:]
        raw_np_crop = raw_np_crop.T
        raw_data_space[:,4*i:4*i+4] = raw_np_crop  #存储有每个片段的通道数据，samples x se*channels
   
        ica = FastICA(n_components=n_com,random_state = 7) 
        components = ica.fit_transform(raw_np_crop) 

        #打印出混合矩阵辅助判断
        print('混合矩阵')
        print(ica.mixing_)
        #画出4通道和4组分时间序列辅助判断
        plt.figure()
        bx1 = plt.subplot(811)
        bx2 = plt.subplot(812)
        bx3 = plt.subplot(813)
        bx4 = plt.subplot(814)
        bx1.plot(times,raw_np_crop[:,0])
        bx2.plot(times,raw_np_crop[:,1])
        bx3.plot(times,raw_np_crop[:,2])
        bx4.plot(times,raw_np_crop[:,3])
        ax1 = plt.subplot(815)
        ax2 = plt.subplot(816)
        ax3 = plt.subplot(817)
        ax4 = plt.subplot(818)
        ax1.plot(times,components[:,0])
        ax2.plot(times,components[:,1])
        ax3.plot(times,components[:,2])
        ax4.plot(times,components[:,3])
        #画出4组分频谱图辅助判断
        PSD = np.zeros((components.shape))
        hz=np.arange(0,200,1)
        for ii in range(components.shape[1]):
            PSD[:,ii] = abs(fft(components[:,ii])/len(components[:,ii])*2)
        plt.figure()
        fx1 = plt.subplot(2,2,1)
        fx2 = plt.subplot(2,2,2)
        fx3 = plt.subplot(2,2,3)
        fx4 = plt.subplot(2,2,4)
        av_PSD = np.zeros((59,4))
        for iii in range(59):
            av_PSD[iii,:] = PSD[iii*5+5:iii*5+10,:].mean(axis=0)
        fx1.plot(hz[1:60],np.log10(av_PSD[:,0]))
        fx2.plot(hz[1:60],np.log10(av_PSD[:,1]))
        fx3.plot(hz[1:60],np.log10(av_PSD[:,2]))
        fx4.plot(hz[1:60],np.log10(av_PSD[:,3]))
        plt.show(block=True)
        #总共出现两张图，都关闭后进入下一组ICs

        components_space[:,n_com*i:n_com*i+n_com] = components
        pattern_space[:,n_com*i:n_com*i+n_com] = ica.mixing_
    print('获得成分空间和模式空间和原始片段时间序列')
    return components_space, pattern_space, raw_data_space

components_space, pattern_space, raw_data_space = get_raw_data()

np.savetxt('../data1/components_space_5.csv', components_space, delimiter=',')
np.savetxt('../data1/pattern_space_5.csv', pattern_space, delimiter=',')
np.savetxt('../data1/raw_data_space_5.csv', raw_data_space, delimiter=',')