import numpy as np

u= np.array([1,2])
v=np.array([3,4])
print(u+v) #vector addition
alpha=5
print(alpha*(u+v)) #scalar multiplication

print(np.dot(u,v)) # dot product
print(np.cross(u,v)) #cross product

print(np.linalg.norm( v-2*u)) # norm (available under linear algebra)
print(np.linalg.norm(u)==0)
print(np.linalg.norm( v-2*u)==1)

a=np.matrix([[1,2],[3,4]]) #matirx (we can also use array instead of matrix call)
b=np.matrix([[9,8],[7,6]])
print(a+b)

print(np.dot(a,u)) #matrix-vector multiplication
print(np.dot(u,a)) 
print(np.dot(a,b)) #matrix multiplication

print(a.T) # transpose of a matirx
print(np.linalg.det(a+b)) #determinant 

print(np.linalg.inv(a)) #inverse
print(np.linalg.inv(b))

x=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(np.diag(x))
print(np.trace(x))
print(np.prod(np.diag(x)))

##for no of rows--> len(x)
##for no of columns --> x.shape[1]
Y=x.shape[1] #no of collumns in a matrix
X=np.linalg.matrix_rank(x) #rank of a matrix
print(Y)
print(X)
nullity= Y - X #finding nullity with the help of rank-nullity theorem
print((nullity))

print(np.multiply(a,b)) #element wise multiplication
print(np.multiply(a,a+b))
print(np.multiply(a+b,a+b))

print(np.outer(a,b)) #each element in a matrix is multiplied with each element of other matrix

print(np.linalg.norm(x,'fro')) #frobenius norm

##Block matirx multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.array([[9, 10], [11, 12]])
D = np.array([[13, 14], [15, 16]])
E = np.array([[17, 18], [19, 20]])
F = np.array([[21, 22], [23, 24]])
G = np.array([[25, 26], [27, 28]])
H = np.array([[29, 30], [31, 32]])
top_left = np.dot(A, E) + np.dot(B, G)
top_right = np.dot(A, F) + np.dot(B, H)
bottom_left = np.dot(C, E) + np.dot(D, G)
bottom_right = np.dot(C, F) + np.dot(D, H)
result = np.block([[top_left, top_right], [bottom_left, bottom_right]])
print(result)