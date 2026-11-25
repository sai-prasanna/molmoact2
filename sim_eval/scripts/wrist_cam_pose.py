import numpy as np

def quat_wxyz_to_R(q):
    w,x,y,z=q; n=np.sqrt(w*w+x*x+y*y+z*z); w,x,y,z=w/n,x/n,y/n,z/n
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])

def R_to_quat_wxyz(R):
    t=np.trace(R)
    if t>0:
        s=np.sqrt(t+1)*2; w=0.25*s; x=(R[2,1]-R[1,2])/s; y=(R[0,2]-R[2,0])/s; z=(R[1,0]-R[0,1])/s
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        s=np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2; w=(R[2,1]-R[1,2])/s; x=0.25*s; y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
    elif R[1,1]>R[2,2]:
        s=np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2; w=(R[0,2]-R[2,0])/s; x=(R[0,1]+R[1,0])/s; y=0.25*s; z=(R[1,2]+R[2,1])/s
    else:
        s=np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2; w=(R[1,0]-R[0,1])/s; x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=0.25*s
    q=np.array([w,x,y,z]); return q/np.linalg.norm(q)

def T(R,p):
    M=np.eye(4); M[:3,:3]=R; M[:3,3]=p; return M

# YAM link_6 frame at home (product of body pos/quat, joints=0)
YAM=[([0,0,0.0631],[0.707107,0,0,0.707107]),
     ([2.5e-05,-0.02,0.0409],[0.5,0.5,0.5,0.5]),
     ([0.264,0,0],[0,1,0,0]),
     ([-0.245,-0.06,0],[1,0,0,0]),
     ([-0.074,-0.0395,2.5e-05],[0.5,-0.5,0.5,0.5]),
     ([0,0.0353,0.0395],[0.707107,-0.707107,0,0])]
Ty6=np.eye(4)
for pos,quat in YAM:
    Ty6=Ty6@T(quat_wxyz_to_R(quat),np.array(pos,float))

# DK1 link6-7 frame at home (product of origin xyz, rpy=0, joints=0)
DK1=[[0,0,0.0615],[0.02,0.0,0.0385],[-0.264,0,0],
     [0.244004,0,0.059971],[0.061868,0,0.0425],[0.032,0,-0.0425]]
Td67=np.eye(4)
for xyz in DK1:
    Td67=Td67@T(np.eye(3),np.array(xyz,float))

# YAM wrist-cam local pose (in link_6 frame)
Cyam=T(quat_wxyz_to_R([0.612372429196013,-0.35355339154618404,-0.3535533966987049,-0.612372438120441]),
       np.array([0,0.09,0.06]))

# C_dk1 = inv(Td67) @ Ty6 @ Cyam   (base frames aligned)
Cdk1=np.linalg.inv(Td67)@Ty6@Cyam
p=Cdk1[:3,3]; q=R_to_quat_wxyz(Cdk1[:3,:3])
print("DK1 wrist-cam local pose in link6-7 frame:")
print("  p =", [round(float(v),6) for v in p])
print("  q =", [round(float(v),6) for v in q])
