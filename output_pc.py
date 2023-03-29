import os
import re

file = open("./output.log")

line = file.readline()

pc_map = dict()

exp_pc = list()


while line:
    # print(line)
    if "(block_id, warp_id)->" in line:
        nums = re.findall(r"\d+\.?\d*", line)
        pc = int(nums[0])
        print(pc)
        vec = []
        vec.append(line)
        line = file.readline()
        nums = re.findall(r"\d+\.?\d*", line)
        exception = int(nums[1])
        vec.append(line)
        if exception != 0:
            exp_pc.append(pc)
            if pc in pc_map.keys():
                pc_map[pc].append(vec)
            else:
                pc_vec = []
                pc_vec.append(vec)
                pc_map[pc] = pc_vec

    if "(block_id, warp_id)=" in line:
        nums = re.findall(r"\d+\.?\d*", line)
        pc = int(nums[0])
        print(pc)
        vec = []
        vec.append(line)
        line = file.readline()
        vec.append(line)
        line = file.readline()
        vec.append(line)
        if pc in pc_map.keys():
            pc_map[pc].append(vec)
        else:
            pc_vec = []
            pc_vec.append(vec)
            pc_map[pc] = pc_vec
    line = file.readline()

file.close()

file = open("output_pc.log", "w")
count1 = 0
count2 = 0
for i in pc_map.keys():
    if i in exp_pc:
        count1 += 1
        file.write("PC_group: %d (Exceptional instruction)\n"%(i))
    else:
        count2 += 1
        file.write("PC_group: %d (Non-exceptional instruction)\n"%(i))
    for v in pc_map[i]:
        for line in v:
            file.write("%s"%(line))
    file.write("\n##################################################\n")
file.write("Summary:\n")
file.write("Total unique PC:%d\n"%(count1 + count2))
file.write("Exceptional PC:%d\n"%(count1))

file.close()
