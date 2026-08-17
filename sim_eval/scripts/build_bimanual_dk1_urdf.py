#!/usr/bin/env python3
"""Generate a bimanual DK1 URDF from the single-arm TRLC-DK1 follower URDF.

Two copies of the follower arm (prefixed ``left_`` / ``right_``) are mounted on
a shared ``bimanual_base`` root link at y = +0.24 / -0.24, both facing +X --
mirroring the YAM bimanual layout (``bimanual_yam_linear_flattened.xml``).

Mesh paths are kept relative ("meshes/visual/..."), so the output URDF must live
in the same directory as the copied ``meshes/`` tree (sim_eval/assets/dk1/).

Run:  python -m sim_eval.scripts.build_bimanual_dk1_urdf
"""
import copy
import os
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "assets", "dk1")
SRC = os.path.join(ASSETS, "TRLC-DK1-Follower.urdf")
OUT = os.path.join(ASSETS, "bimanual_dk1.urdf")

ARM_Y_OFFSET = 0.24  # left at +0.24, right at -0.24 (matches YAM flattened)


def _prefix_refs(elem, prefix):
    """Prefix every link/joint name reference inside a cloned <link>/<joint>."""
    if elem.tag in ("link", "joint") and "name" in elem.attrib:
        elem.set("name", prefix + elem.get("name"))
    for child in elem.iter():
        if child.tag in ("parent", "child") and "link" in child.attrib:
            child.set("link", prefix + child.get("link"))


def build():
    src_root = ET.parse(SRC).getroot()
    src_links = [e for e in src_root if e.tag == "link"]
    src_joints = [e for e in src_root if e.tag == "joint"]

    robot = ET.Element("robot", {"name": "bimanual_dk1"})

    # Shared root link.
    base = ET.SubElement(robot, "link", {"name": "bimanual_base"})
    inertial = ET.SubElement(base, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": "1.0"})
    ET.SubElement(inertial, "inertia",
                  {"ixx": "0.001", "ixy": "0", "ixz": "0",
                   "iyy": "0.001", "iyz": "0", "izz": "0.001"})

    for side, y in (("left_", ARM_Y_OFFSET), ("right_", -ARM_Y_OFFSET)):
        # Fixed mount: bimanual_base -> <side>base_link
        mount = ET.SubElement(robot, "joint",
                              {"name": f"{side}mount", "type": "fixed"})
        ET.SubElement(mount, "origin", {"xyz": f"0 {y} 0", "rpy": "0 0 0"})
        ET.SubElement(mount, "parent", {"link": "bimanual_base"})
        ET.SubElement(mount, "child", {"link": f"{side}base_link"})

        for link in src_links:
            clone = copy.deepcopy(link)
            _prefix_refs(clone, side)
            robot.append(clone)
        for joint in src_joints:
            clone = copy.deepcopy(joint)
            _prefix_refs(clone, side)
            robot.append(clone)

        # NOTE: a previous version inserted a fixed `gripper_mount` link here to
        # apply the ~5.6 mm grasp-center offset correction M. That FROZE the
        # gripper (inserting a link between link6-7 and the prismatic finger
        # joints breaks physx actuation of those joints — verified: with the
        # mount the fingers never move, with or without mass). The offset was
        # minor and not the grasp blocker, so the mount is intentionally NOT
        # inserted: the fingers + tool0 stay direct children of link6-7.

    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(OUT, encoding="utf-8", xml_declaration=True)
    print(f"wrote {OUT}")
    # quick stats
    n_rev = sum(1 for j in robot if j.tag == "joint" and j.get("type") == "revolute")
    n_pri = sum(1 for j in robot if j.tag == "joint" and j.get("type") == "prismatic")
    print(f"  revolute joints: {n_rev}  prismatic joints: {n_pri}")


if __name__ == "__main__":
    build()
