from autograd import jacobian
import autograd.numpy as np
def f_test1(inp):
    return 100*np.power(inp[1]-np.power(inp[0],2),2)+np.power(1-inp[0],2)+100*np.power(inp[2]-np.power(inp[1],2),2)
def f_test2(inp):
    return 100*np.power(inp[1]-np.power(inp[0],2),2)+np.power(1-inp[0],2)
def f_test3(inp):
    return 1+np.power(1-inp[0],2)
def f_test4(inp):
    return 1+np.power(np.array([1])-np.sin(inp[0]),2)
def f_test5(inp):
    return np.array([1])+np.power(np.array([1])-np.sin(inp[0]),2)+np.power(np.array([1])-np.cos(inp[1]),2)
f_test=[f_test1,f_test2,f_test3,f_test4,f_test5]
inp=[np.array([1.0001,1.0001,1.000001]),
     np.array([1.2,1.2]),
     np.array([1.2]),
     np.array([1.2]),
     np.array([1.2,1.2])]
f_str=['(1-x1)^2+(x1-x2^2)^2+(x2-x3^2)^2',
       '(1-x1)^2+(x1-x2^2)^2',
       '(1-x1)^2',
       '1+(1-sin(x1))^2',
       '1+(1-sin(x1))^2+(1-cos(x2))^2']

def op(function,begin,eps=0.000000001,rou=0.4,sigma=0.5):
    grad_f=jacobian(function)
    x=begin
    d=-grad_f(x).reshape([-1])
    #print(d)
    term=0
    while (np.sum(np.power(d,2))>eps):
        if term%1000==0:
            print('term:'+str(term))
        term+=1
        zero_grad=float(-np.sum(np.power(d,2)))
        #a=find_a(function,x,d)
        a=1
        #print(a)
        a1=0
        a2=a
        alpha=(a1+a2)/2
        phi_1=float(function(x))
        phi_1_star=zero_grad
        while True:
            #print('outest loop alpha:'+str(alpha))
            phi=float(function(x+alpha*d))
            while(phi>(function(x)+rou*alpha*zero_grad)):
                #print('phi:'+str(phi))
                #print('left:'+str(function(x)+rou*alpha*zero_grad))
                if alpha<0:
                    raise ValueError
                a2,alpha=alpha,a1+0.5*np.power(a1-alpha,2)*phi_1_star/(phi_1-phi-(a1-alpha)*phi_1_star)
               # print(alpha)
                phi=float(function(x+alpha*d))
                #print('a2:'+str(a2))
                #print('a1:'+str(a1))
                #print('first loop alpha:'+str(alpha))
                #print('phi:'+str(phi))
                if a1==a2:
                    alpha=a1
                    break
            
            
            phi_star=float(np.dot(grad_f(x+alpha*d),d))
            #print('phi star:'+str(phi_star))
            #print('phi 1 star:'+str(phi_1_star))
            #print('a2 grad:'+str(np.dot(grad_f(x+a2*d),d)))
            #print('right:'+str(sigma*zero_grad))
            if (phi_star>=sigma*zero_grad):
                break
            else:
                a1,alpha=alpha,alpha-(a1-alpha)*phi_star/(phi_1_star-phi_star)
                if alpha<a1:
                    alpha=(a1+a2)/2
                #print('phi star:'+str(phi_star))
                #print('phi 1 star:'+str(phi_1_star))
                #print('a2 grad:'+str(np.dot(grad_f(x+a2*d),d)))
                #print('right:'+str(sigma*zero_grad))
                phi_1=phi
                phi_1_star=phi_star
                #print('a2:'+str(a2))
                #print('a1:'+str(a1))
                #print('second loop alpha:'+str(alpha))
            if a1==a2:
                alpha=a1
                break
        #print(alpha)
        #print(d)
        x_old=x
        x=x+alpha*d.reshape([-1])
        if term%1000==0:
            print('x:'+str(x))
            print('f(x):'+str(function(x)))
        if np.sum(np.power(x-x_old,2))<0.000000000000000001:
            return 'gg'
        d=-grad_f(x).reshape([-1])
    return (x,function(x))
for i in range(5):
    print('function:'+f_str[i])
    print('start:'+str(inp[i]))
    print('f(star):'+str(f_test[i](inp[i])))
    best=op(f_test[i],inp[i])
    print('op point:'+str(best[0]))
    print('f(op):'+str(best[1]))
    
