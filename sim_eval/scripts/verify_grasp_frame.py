"""Convention-free physical comparison via FK.

closing axis  = gripper prismatic slide direction, in base frame
approach axis = wrist -> TCP reach direction, in base frame
Computed for YAM and DK1 at identical arm joint angles; report angular diff.
If these stay small across configs, identity joint map preserves physical grasp
orientation -> zero-shot transfer is sound.
"""
import numpy as np

def q2R(q):
    w,x,y,z=q; n=np.sqrt(w*w+x*x+y*y+z*z); w,x,y,z=w/n,x/n,y/n,z/n
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
def axR(axis,a):
    u=np.array(axis,float); u/=np.linalg.norm(u); x,y,z=u; c,s=np.cos(a),np.sin(a); C=1-c
    return np.array([[c+x*x*C,x*y*C-z*s,x*z*C+y*s],
                     [y*x*C+z*s,c+y*y*C,y*z*C-x*s],
                     [z*x*C-y*s,z*y*C+x*s,c+z*z*C]])
def Tm(R,p):
    M=np.eye(4); M[:3,:3]=R; M[:3,3]=p; return M

YAM=[([0,0,0.0631],[0.707107,0,0,0.707107],[0,0,1]),
     ([2.5e-05,-0.02,0.0409],[0.5,0.5,0.5,0.5],[0,0,1]),
     ([0.264,0,0],[0,1,0,0],[0,0,1]),
     ([-0.245,-0.06,0],[1,0,0,0],[0,0,1]),
     ([-0.074,-0.0395,2.5e-05],[0.5,-0.5,0.5,0.5],[0,0,1]),
     ([0,0.0353,0.0395],[0.707107,-0.707107,0,0],[0,0,1])]
def yam_link6_R(q):
    M=np.eye(4)
    for (pos,quat,axis),qi in zip(YAM,q):
        M=M@Tm(q2R(quat),np.array(pos,float))@Tm(axR(axis,qi),np.zeros(3))
    return M[:3,:3]
# YAM finger slide axis in link_6 frame: finger body quat * local z
YAM_CLOSE = q2R([0.707107,0,0.707107,0])@np.array([0,0,1.])
YAM_APPR  = np.array([0,0,1.])  # grasp_site offset [0,0,0.1347]

DK1=[([0,0,0.0615],[0,0,1]),([0.02,0,0.0385],[0,1,0]),([-0.264,0,0],[0,-1,0]),
     ([0.244004,0,0.059971],[0,-1,0]),([0.061868,0,0.0425],[0,0,-1]),([0.032,0,-0.0425],[1,0,0])]
def dk1_link67_R(q):
    M=np.eye(4)
    for (xyz,axis),qi in zip(DK1,q):
        M=M@Tm(np.eye(3),np.array(xyz,float))@Tm(axR(axis,qi),np.zeros(3))
    return M[:3,:3]
DK1_CLOSE = np.array([0,1,0.])  # gripper_left axis
DK1_APPR  = np.array([1,0,0.])  # tool0 offset [0.158,0,0]

CONFIGS={"home":[0,0,0,0,0,0],"reach1":[0.4,0.6,0.5,-0.3,0.2,0.1],
         "reach2":[-0.5,0.9,1.0,0.5,-0.4,0.3],"reach3":[0.2,1.2,0.3,-0.8,0.7,-0.6],
         "reach4":[1.0,0.3,-0.2,1.2,1.0,-1.5]}
def ang(a,b):
    a=a/np.linalg.norm(a); b=b/np.linalg.norm(b)
    return np.degrees(np.arccos(np.clip(a@b,-1,1)))
print(f"{'config':8s} {'closing-axis Δ°':>16s} {'approach-axis Δ°':>18s}")
for n,arm in CONFIGS.items():
    Ry=yam_link6_R(arm); Rd=dk1_link67_R(arm)
    dc=ang(Ry@YAM_CLOSE, Rd@DK1_CLOSE)
    da=ang(Ry@YAM_APPR,  Rd@DK1_APPR)
    print(f"{n:8s} {dc:16.3f} {da:18.3f}")
