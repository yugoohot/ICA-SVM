import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

def train(components_trait):
    np.random.shuffle(components_trait)

    #train_data = components_trait[:236,:]
    #test_data = components_trait[236:,:]
    #train_label = components_label[:236]
    #test_label = components_label[236:]
    svm_model = SVC(kernel = 'linear')
    score = (cross_val_score(svm_model,components_trait[:,1:25],components_trait[:,0],cv=10,n_jobs=-1))
    acc = score.mean()
    print('平均精度：')
    print(acc)
    print('详细精度：')
    print(score)
    np.savetxt('./score.csv', score, delimiter=',')
'''    svm_model.fit(train_data,train_label)

    pre_train = svm_model.predict(train_data)
    accuracy_train = accuracy_score(train_label,pre_train)
    print('训练精度：')
    print(accuracy_train)

    pre_test = svm_model.predict(test_data)
    accuracy_test = accuracy_score(test_label,pre_test)
    print('测试精度：')
    print(accuracy_test)
    return svm_model'''


f_trait_space = r'../data/trait_space.csv'
with open(f_trait_space,encoding = 'utf-8') as pp:
    trait_space = np.loadtxt(pp,str,delimiter = ",",comments='%')
trait_space=np.array(trait_space,dtype='float64')

'''f_components_label = r'../data/components_label_1+2.csv'
with open(f_components_label,encoding = 'utf-8') as pp:
    components_label = np.loadtxt(pp,str,delimiter = ",",comments='%')
components_label = np.array(components_label,dtype='int32')'''
bool_matrix=trait_space==2
trait_space[bool_matrix] = 0
model = train(trait_space.T,)
