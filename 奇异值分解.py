import numpy as np

#特征值分解
A = np.random.randint(-10,10,(4,4))
print(A)
b = np.dot(A.T,A)
print(b)

vals,vecs = np.linalg.eig(b)
print(vals)
print(vecs)

#将特征值转换为矩阵
vals2vec = np.diag(vals)
print(vals2vec)
print(b)
print(np.dot(np.dot(vecs,vals2vec),vecs.T)) #与b相同
print(np.dot(np.dot(vecs.T,vals2vec),vecs)) #与b不同
#故验证了使用np中的eig分解为A=P*B*PT 而不是A=QT*B*Q，其中P=vecs，

#然后再来看使用np中的eig分解出来的vec中行向量是特征向量还是列向量是特征向量，只需验证：A*vecs[0] = vals[0]*vecs[0]
print(np.dot(b,vecs[0]))
print(vals[0]*vecs[0])

print(np.dot(b,vecs[:,0]))
print(vals[0]*vecs[:,0])
#后者两个是相等的，故使用np中的eig分解出的vecs的列向量是特征向量。

#验证P是单位正交阵
print(np.dot(vecs.T,vecs))
print(np.dot(vecs,vecs.T))
# 可以看到除对角元外其他都是非常小的数


##奇异值分解
a = np.random.randint(-10,10,(4,3)).astype(np.float32)

u,s,v = np.linalg.svd(a)
print(u.shape)
print(s.shape)
print(v.shape)

print(np.allclose(a,np.dot(u[:,:3]*s,v)))
print(u[:,:3])
print(u)
print(s)
# print(a)
# print(np.dot(u[:,:3]*s,v))
# #将s转化为奇异值矩阵

smat = np.zeros((4,3))
smat[:3, :3] = np.diag(s)
print(smat)

print(np.allclose(a,np.dot(u,np.dot(smat,v))))


