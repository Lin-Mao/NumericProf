import os
import re

file = open("./output.log")

line = file.readline()

pc_map = dict()

exp_pc = list()

while line:
    # print(line)
    if "(block_id, warp_id)=(0, (1,0,0), 0)" in line:
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

warp_data = dict()
for pc in pc_map.keys():
    vec = pc_map[pc][0]
    nums1 = vec[1].split()
    nums2 = vec[2].split()
    for i in range(32):
        if i not in warp_data.keys():
            warp_data[i] = list()
        warp_data[i].append((pc, nums1[i+1], nums2[i+1]))

print(warp_data)

file = open("ttt.json", 'w')
file.write("[")
for lane in warp_data.keys():
    count = 0
    if lane != 29:
        continue
    for i in warp_data[lane]:
        file.write("{\"pid\":0,\"tid\":"+ str(lane) +",\"ts\":" + str(count) + ",\"ph\":\"X\",\"cat\":\"K\",\"name\":\"" \
               "ITER" + str(count) + "\",\"dur\":1,\"args\":{\"PC\":" + str(i[0]) +",\"Exponent\":\""+ i[2] +"\"}},\n")
        count += 1
    meta1 = "{\"pid\":0,\"tid\":"+ str(lane) +",\"ts\":"+ str(lane) +",\"ph\":\"M\",\"cat\":\"__metadata\",\"name\":\"" \
             "thread_name\",\"args\":{\"name\":\"lane" + str(lane) + "\"}},"
    file.write(meta1 + "\n")
    meta2 = "{\"pid\":0,\"tid\":0,\"ts\":0,\"ph\":\"M\",\"cat\":\"__metadata\",\"name\":\"" \
                    "process_name\",\"args\":{\"name\":\"Variables\"}},"
    file.write(meta2 + "\n")
file.write("]")


  

