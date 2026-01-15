# Order: [thumb, index, middle, ring, little]
# 0: Rest
# 1-17: Exercise B (E1)
# 18-40: Exercise C (E2)
# 41-49: Exercise D (E3)
# Note: descriptions are taken directly from Ninapro DB2 exercise descriptions

DB2_5BIT = {0: [0,0,0,0,0]}

# Exercise B, E1
B = {
    1:  [1,0,0,0,0],  # Thumb up
    2:  [1,1,1,1,1],  # Extension of index+middle, flexion of others
    3:  [1,1,1,1,1],  # Flexion of ring+little, extension of others
    4:  [1,0,0,0,1],  # Thumb opposing base of little finger
    5:  [1,1,1,1,1],  # Abduction of all fingers
    6:  [1,1,1,1,1],  # Fingers flexed together in fist
    7:  [1,1,1,1,1],  # Pointing index (index ext, others flex)
    8:  [1,1,1,1,1],  # Adduction of extended fingers
    9:  [0,0,0,0,0],  # Wrist supination (axis: middle finger)
    10: [0,0,0,0,0],  # Wrist pronation (axis: middle finger)
    11: [0,0,0,0,0],  # Wrist supination (axis: little finger)
    12: [0,0,0,0,0],  # Wrist pronation (axis: little finger)
    13: [0,0,0,0,0],  # Wrist flexion
    14: [0,0,0,0,0],  # Wrist extension
    15: [0,0,0,0,0],  # Wrist radial deviation
    16: [0,0,0,0,0],  # Wrist ulnar deviation
    17: [0,0,0,0,0],  # Wrist extension with closed hand
}
DB2_5BIT.update(B)

# Exercise C, E2
C = {
    18: [1,1,1,1,1],  # 1 Large diameter grasp
    19: [1,1,1,1,1],  # 2 Small diameter grasp (power grip)
    20: [0,1,1,1,1],  # 3 Fixed hook grasp (no thumb)
    21: [1,1,1,1,1],  # 4 Index finger extension grasp
    22: [1,1,1,1,1],  # 5 Medium wrap
    23: [1,1,1,1,0],  # 6 Ring grasp
    24: [0,1,1,1,1],  # 7 Prismatic four fingers grasp
    25: [1,1,1,0,0],  # 8 Stick grasp
    26: [1,1,1,0,0],  # 9 Writing tripod grasp
    27: [1,1,1,1,1],  # 10 Power sphere grasp
    28: [1,1,1,0,0],  # 11 Three finger sphere grasp
    29: [1,1,1,0,0],  # 12 Precision sphere grasp
    30: [1,1,1,0,0],  # 13 Tripod grasp
    31: [1,1,0,0,0],  # 14 Prismatic pinch grasp
    32: [1,1,0,0,0],  # 15 Tip pinch grasp
    33: [1,1,1,1,0],  # 16 Quadpod grasp
    34: [1,1,0,0,0],  # 17 Lateral grasp (key pinch)
    35: [1,1,1,0,0],  # 18 Parallel extension grasp
    36: [1,1,1,1,1],  # 19 Extension type grasp
    37: [1,1,1,1,1],  # 20 Power disk grasp
    38: [1,1,1,0,0],  # 21 Open bottle with a tripod grasp
    39: [1,1,1,0,0],  # 22 Turn a screw (screwdriver / stick grasp)
    40: [1,1,1,0,0]   # 23 Knife cutting (index extension grasp)
}
DB2_5BIT.update(C)

# Exercise D, E3
D = {
    41: [0,0,0,0,1],  # Flex little
    42: [0,0,0,1,0],  # Flex ring
    43: [0,0,1,0,0],  # Flex middle
    44: [0,1,0,0,0],  # Flex index
    45: [1,0,0,0,0],  # Abduction of thumb
    46: [1,0,0,0,0],  # Flex thumb
    47: [0,1,0,0,1],  # Flex index + little
    48: [0,0,1,1,0],  # Flex ring + middle
    49: [1,1,0,0,0],  # Flex index + thumb
}
DB2_5BIT.update(D)

# Function to map gesture ID to 5-bit finger activation vector
def gesture_to_5bit(gid: int, strict: bool = False):
    gid = int(gid)
    if strict and gid not in DB2_5BIT:
        raise KeyError(f"Unmapped gesture id: {gid}")
    return DB2_5BIT.get(gid, [0,0,0,0,0])
