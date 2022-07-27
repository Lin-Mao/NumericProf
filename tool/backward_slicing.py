import json

backward_slicing_blocks = dict()


def search_in_block(prev_bbid: int, bbid: int, block: list, start_point: int, val: str, insts_map: dict) -> str:
    keep = list()
    for i in range(start_point, -1, -1):
        inst,_ = insts_map[block[i]]
        if val not in inst:
            continue
        else:
            inst = inst.split()
            if val in inst[1]:
                if "MOV" not in inst[0]:
                    keep.append(block[i])
                else:
                    keep.append(block[i])
                    val = inst[2]
    # print(keep)
    if prev_bbid not in backward_slicing_blocks:
        backward_slicing_blocks[prev_bbid] = []
    backward_slicing_blocks[prev_bbid].append((bbid, keep))
    return val


def traverse_slicing_cfg(prev_bbid: int, bbid: int, start_point: int, basic_blocks: dict, val: str, insts_map: dict,
                     b_slicing: dict) -> list:

    block = basic_blocks[bbid]
    val = search_in_block(prev_bbid, bbid, block, start_point, val, insts_map)
    if bbid not in b_slicing.keys():
        return
    for i in b_slicing[bbid]:
        block = basic_blocks[i]
        start_point = len(block)-1
        traverse_slicing_cfg(bbid, i, start_point, basic_blocks, val, insts_map, b_slicing)


def backward_slicing(pc: int, basic_blocks: dict, insts_map: dict, reverse_cfg: dict) -> list:
    inst, bb = insts_map[pc]

    b_slicing = dict()
    queue = list()
    queue.append(bb)
    while queue:
        b = queue.pop(0)
        if b not in reverse_cfg:
            continue
        for i in reverse_cfg[b]:
            queue.append(i)
            if b not in b_slicing.keys():
                b_slicing[b] = []
            b_slicing[b].append(i)
    print(b_slicing)

    val = ""
    if "STG" in inst:
        inst = inst.split()
        val = inst[-1]
        print(val)

    block = basic_blocks[bb]
    print(block)
    index = block.index(pc)

    traverse_slicing_cfg(-1, bb, index, basic_blocks, "R5", insts_map, b_slicing)


def main():
    sass_file = "C:/Users/mao/Desktop/json/a.sass"
    json_file = "C:/Users/mao/Desktop/json/div.json"

    file = open(sass_file)
    line = file.readline()
    insts_map = dict()
    while line:
        inst = line.strip().split('$ ')
        insts_map[int(inst[2])] = inst[1].rstrip(';').rstrip()
        line = file.readline()

    with open(json_file, 'r') as f_json:
        load_json = json.load(f_json)
    kernel = list(load_json.values())[0]
    blocks = kernel["blocks"]

    cfg = dict()
    for bb in blocks:
        dst = list()
        bb_id = int(bb["id"])
        targets = bb["targets"]
        for t in targets:
            dst.append(int(t["id"]))
        cfg[bb_id] = dst
    print(cfg)

    reverse_cfg = dict()
    for bb in cfg.keys():
        for i in cfg[bb]:
            if i not in reverse_cfg.keys():
                reverse_cfg[i] = []
            reverse_cfg[i].append(bb)
    print(reverse_cfg)

    basic_blocks = dict()
    for bb in blocks:
        dst = list()
        bb_id = int(bb["id"])
        insts = bb["insts"]
        for i in insts:
            pc = int(i["pc"])
            dst.append(pc)
            insts_map[pc] = (insts_map[pc], bb_id)
        basic_blocks[bb_id] = dst

    print("basic_blocks:", basic_blocks)
    print("insts_map:", insts_map)

    pc =  208
    backward_slicing(pc, basic_blocks, insts_map, reverse_cfg)


if __name__ == "__main__":
    main()
    print(backward_slicing_blocks)