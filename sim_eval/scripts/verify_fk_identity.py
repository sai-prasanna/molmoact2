"""Compare YAM (MJCF) vs DK1 (URDF) single-arm forward kinematics.

Pure numpy. Tests whether the arm joint mapping is identity by comparing two
frame-invariant quantities across random joint configs:
  1. geodesic angle of R(q) R(home)^T   (invariant under constant base rotation)
  2. ||p_tcp(q) - p_tcp(home)||          (TCP displacement magnitude)
"""
import numpy as np

def quat_wxyz_to_R(q):
    w, x, y, z = q
    n = np.sqrt(w*w+x*x+y*y+z*z); w,x,y,z = w/n,x/n,y/n,z/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])

def rpy_to_R(r, p, y):
    cr,sr=np.cos(r),np.sin(r); cp,sp=np.cos(p),np.sin(p); cy,sy=np.cos(y),np.sin(y)
    Rz=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    Ry=np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rx=np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    return Rz@Ry@Rx

def axis_angle_R(axis, ang):
    a=np.asarray(axis,float); a=a/np.linalg.norm(a)
    x,y,z=a; c,s=np.cos(ang),np.sin(ang); C=1-c
    return np.array([
        [c+x*x*C,   x*y*C-z*s, x*z*C+y*s],
        [y*x*C+z*s, c+y*y*C,   y*z*C-x*s],
        [z*x*C-y*s, z*y*C+x*s, c+z*z*C],
    ])

def T(R, p):
    M=np.eye(4); M[:3,:3]=R; M[:3,3]=p; return M

# ---- YAM single arm (from yam.xml). Each entry: (pos, quat_wxyz, joint_axis) ----
YAM = [
    ([0,0,0.0631],            [0.707107,0,0,0.707107],     [0,0,1]),  # link_1 / joint1
    ([2.5e-05,-0.02,0.0409],  [0.5,0.5,0.5,0.5],           [0,0,1]),  # link_2 / joint2
    ([0.264,0,0],             [0,1,0,0],                   [0,0,1]),  # link_3 / joint3
    ([-0.245,-0.06,0],        [1,0,0,0],                   [0,0,1]),  # link_4 / joint4 (no quat)
    ([-0.074,-0.0395,2.5e-05],[0.5,-0.5,0.5,0.5],          [0,0,1]),  # link_5 / joint5
    ([0,0.0353,0.0395],       [0.707107,-0.707107,0,0],    [0,0,1]),  # link_6 / joint6
]
YAM_TCP_local = np.array([0,0,0.1347])   # grasp_site in link_6 frame (position only)

def yam_fk(q):
    M=np.eye(4)
    for (pos,quat,axis),qi in zip(YAM,q):
        M = M @ T(quat_wxyz_to_R(quat), np.array(pos,float)) @ T(axis_angle_R(axis,qi), np.zeros(3))
    R=M[:3,:3]; p=M[:3,:3]@YAM_TCP_local + M[:3,3]
    return R, p

# ---- DK1 (from URDF). Each: (origin_xyz, origin_rpy, axis) ----
DK1 = [
    ([0,0,0.0615],          [0,0,0], [0,0,1]),
    ([0.02,0.0,0.0385],     [0,0,0], [0,1,0]),
    ([-0.264,0,0],          [0,0,0], [0,-1,0]),
    ([0.244004,0,0.059971], [0,0,0], [0,-1,0]),
    ([0.061868,0,0.0425],   [0,0,0], [0,0,-1]),
    ([0.032,0,-0.0425],     [0,0,0], [1,0,0]),
]
DK1_TCP_origin = ([0.158,0,0], [-1.5707963267948966,0,-1.5707963267948966])  # gripper_tool0

def dk1_fk(q):
    M=np.eye(4)
    for (xyz,rpy,axis),qi in zip(DK1,q):
        M = M @ T(rpy_to_R(*rpy), np.array(xyz,float)) @ T(axis_angle_R(axis,qi), np.zeros(3))
    xyz,rpy=DK1_TCP_origin
    M = M @ T(rpy_to_R(*rpy), np.array(xyz,float))
    return M[:3,:3], M[:3,3]

def geo_angle(Ra,Rb):
    Rr=Ra@Rb.T
    c=(np.trace(Rr)-1)/2
    return np.degrees(np.arccos(np.clip(c,-1,1)))

rng=np.random.default_rng(0)
Ry0,py0=yam_fk(np.zeros(6)); Rd0,pd0=dk1_fk(np.zeros(6))
print("home YAM tcp:", np.round(py0,4), " DK1 tcp:", np.round(pd0,4))

N=2000
max_ang_err=0.0; max_pos_err=0.0
for _ in range(N):
    q=rng.uniform(-1.0,1.0,6)
    Ry,py=yam_fk(q); Rd,pd=dk1_fk(q)
    ang_y=geo_angle(Ry,Ry0); ang_d=geo_angle(Rd,Rd0)
    dist_y=np.linalg.norm(py-py0); dist_d=np.linalg.norm(pd-pd0)
    max_ang_err=max(max_ang_err, abs(ang_y-ang_d))
    max_pos_err=max(max_pos_err, abs(dist_y-dist_d))
print(f"IDENTITY mapping  q_dk1 = q_yam:")
print(f"  max |orientation-angle diff|  = {max_ang_err:.4f} deg")
print(f"  max |tcp-displacement diff|   = {max_pos_err*1000:.4f} mm  (over {N} random configs)")

# Per-joint single-axis sweep: rotate only joint i, report angle match
print("\nPer-joint single-joint sweeps (only joint i moved by 0.7 rad):")
for i in range(6):
    q=np.zeros(6); q[i]=0.7
    Ry,py=yam_fk(q); Rd,pd=dk1_fk(q)
    print(f"  j{i+1}: YAM dAng={geo_angle(Ry,Ry0):6.2f}  DK1 dAng={geo_angle(Rd,Rd0):6.2f} | "
          f"YAM dPos={np.linalg.norm(py-py0)*1000:6.1f}mm DK1 dPos={np.linalg.norm(pd-pd0)*1000:6.1f}mm")
