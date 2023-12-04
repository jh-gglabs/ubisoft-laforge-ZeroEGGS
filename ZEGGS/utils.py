import numpy as np
from scipy import interpolate

from anim import bvh, quat


def change_bvh(filename, savename, order=None, fps=None, pace=1.0, center=False):
    anim_data = bvh.load(filename)
    output = anim_data.copy()

    if order is not None:
        output["order"] = order
        rotations = quat.unroll(quat.from_euler(np.radians(anim_data['rotations']), order=anim_data['order']))
        output["rotations"] = np.degrees(quat.to_euler(rotations, order=output["order"]))
    if pace is not None or fps is not None:
        if fps is None:
            fps = 1.0 / anim_data["frametime"]
        positions = anim_data['positions']
        rotations = quat.unroll(quat.from_euler(np.radians(anim_data['rotations']), order=anim_data['order']))
        nframes = positions.shape[0]
        nbones = positions.shape[1]
        original_times = np.linspace(0, nframes - 1, nframes)
        sample_times = np.linspace(
            0, nframes - 1, int(pace * (nframes * (fps * anim_data["frametime"]) - 1))
        )
        output["positions"] = interpolate.griddata(original_times, output["positions"].reshape([nframes, -1]),
                                                   sample_times, method='cubic').reshape([len(sample_times), nbones, 3])
        rotations = interpolate.griddata(original_times, rotations.reshape([nframes, -1]),
                                         sample_times, method='cubic').reshape([len(sample_times), nbones, 4])
        rotations = quat.normalize(rotations)
        output["rotations"] = np.degrees(quat.to_euler(rotations, order=output["order"]))
        output["frametime"] = 1.0 / fps

    if center:
        lrot = quat.from_euler(np.radians(output["rotations"]), output["order"])
        offset_pos = output["positions"][0:1, 0:1].copy() * np.array([1, 0, 1])
        offset_rot = lrot[0:1, 0:1].copy() * np.array([1, 0, 1, 0])

        root_pos = quat.mul_vec(quat.inv(offset_rot), output["positions"][:, 0:1] - offset_pos)
        output["positions"][:, 0:1] = quat.mul_vec(quat.inv(offset_rot),
                                                   output["positions"][:, 0:1] - offset_pos)
        output["rotations"][:, 0:1] = np.degrees(
            quat.to_euler(quat.mul(quat.inv(offset_rot), lrot[:, 0:1]), order=output["order"]))
    bvh.save(savename, output)


def write_bvh(
        
        V_root_pos,
        V_root_rot,
        V_lpos,
        V_lrot,
        parents,
        names,
        order,
        dt,
        filename=None,
        start_position=None,
        start_rotation=None,
):
    if start_position is not None and start_rotation is not None:
        offset_pos = V_root_pos[0:1].copy()
        offset_rot = V_root_rot[0:1].copy()

        V_root_pos = quat.mul_vec(quat.inv(offset_rot), V_root_pos - offset_pos)
        V_root_rot = quat.mul(quat.inv(offset_rot), V_root_rot)
        V_root_pos = (
                quat.mul_vec(start_rotation[np.newaxis], V_root_pos) + start_position[np.newaxis]
        )
        V_root_rot = quat.mul(start_rotation[np.newaxis], V_root_rot)

    V_lpos = V_lpos.copy()
    V_lrot = V_lrot.copy()
    V_lpos[:, 0] = quat.mul_vec(V_root_rot, V_lpos[:, 0]) + V_root_pos
    V_lrot[:, 0] = quat.mul(V_root_rot, V_lrot[:, 0])

    bvh_info = dict(
            order=order,
            offsets=V_lpos[0],
            names=names,
            frametime=dt,
            parents=parents,
            positions=V_lpos,
            rotations=np.degrees(quat.to_euler(V_lrot, order=order)),
    )

    if filename is not None:
        bvh.save(filename, bvh_info)

    return bvh_info


def space2t(test_sentence):
    res_str = ''
    for i in range(0, len(test_sentence), 2):
        if test_sentence[i:i + 2] == '  ':
            res_str += '\t'
        else:
            res_str += test_sentence[i:]
            break
    return res_str


def convert_ipose_to_tpose(bvh_file_path, out_file_path):
    """ Conversion I-pose to T-pose
    Ref: https://github.com/YoungSeng/UnifiedGesture/blob/master/retargeting/datasets/process_bvh.py
    """
    motion = []
    divide = 1 # FPS
    with open(out_file_path, 'w') as output:
        with open(bvh_file_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if i == 3:
                    line = line.strip().split(' ')
                    x = float(line[-3])
                    y = float(line[-2])
                    z = float(line[-1])
                    # print('root', x, y, z)
                    line[-3] = '0.0'
                    line[-1] = '0.0'
                    line = '  ' + ' '.join(line) + '\n'
                    line = space2t(line)
                
                if ((51 <= i <= 75 or 87 <= i <= 99 or 111 <= i <= 123 or 135 <= i <= 147 or 159 <= i <= 171) and i % 4 ==3):     # RightArm OFFSET
                    line = line.strip('\n').split(' ')
                    x = float(line[-3])
                    y = float(line[-2])
                    z = float(line[-1])
                    line[-3] = str(0.0 - y)
                    line[-2] = str(x)
                    line[-1] = str(z)
                    line = ' '.join(line) + '\n'
                    line = space2t(line)
                if ((209 <= i <= 233 or 245 <= i <= 257 or 269 <= i <= 281 or 293 <= i <= 305 or 317 <= i <= 329) and i % 4 ==1):     # RightArm OFFSET
                    line = line.strip('\n').split(' ')
                    x = float(line[-3])
                    y = float(line[-2])
                    z = float(line[-1])
                    line[-3] = str(y)
                    line[-2] = str(0.0 - x)
                    line[-1] = str(z)
                    line = ' '.join(line) + '\n'
                    line = space2t(line)
                if i < 461:
                    output.write(space2t(line))
                    continue
                if i == 461:
                    frames = int(line.strip().split(' ')[-1])  # \t for testing and ' ' for training
                    print(frames)
                    output.write('Frames: ' + str((frames + 1) // divide) + '\n')
                    continue
                if i == 462:
                    fps = float(line.strip().split(' ')[-1])  #
                    print(fps)
                    if divide == 2:
                        output.write('Frame Time: ' + str(1 / 30.0) + '\n')
                    elif divide == 1:
                        output.write('Frame Time: ' + str(1 / 60.0) + '\n')
                    continue
                else:
                    motion.append(line)
        
        if len(motion) != frames:
            print(len(motion), '/', frames)
            motion = motion[:frames]

        motion = motion[::divide]
        if divide == 2:
            assert len(motion) == (frames + 1) // divide     # fps = 1/30.0
        elif divide == 1:
            assert len(motion) == (frames) // divide
        for i in motion:
            i = i.strip().split(' ')

            z = float(i[30])
            y = float(i[31])
            x = float(i[32])
            i[30] = str(z - 90.0)
            i[31] = str(x)
            i[32] = str(0.0 - y)

            # print(z, y, x, i[30], i[31], i[32])      # 92.418716 -0.394881 9.707915

            for j in range(11 * 3, 35 * 3, 3):
                z = float(i[j])
                y = float(i[j+1])
                x = float(i[j+2])
                i[j+1] = str(x)
                i[j+2] = str(0.0 - y)
                # print(i[j], i[j+1], i[j+2])

            z = float(i[36 * 3])
            y = float(i[36 * 3 + 1])
            x = float(i[36 * 3 + 2])
            i[36 * 3] = str(z + 90.0)
            i[36 * 3 + 1] = str(0.0 - x)
            i[36 * 3 + 2] = str(y)

            # print(z, y, x, i[36 * 3], i[36 * 3 + 1], i[36 * 3 + 2])     # -90.370065 0.051974 5.208683

            for j in range(37 * 3, 61 * 3, 3):
                y = float(i[j+1])
                x = float(i[j+2])
                i[j+1] = str(0.0 - x)
                i[j+2] = str(y)

            i = ' '.join(i) + '\n'
            output.write(i)


def convert_to_mixamo_bone_structure(bvh_dict):
    remove_bones = [
        "Spine3", "Neck1",
        "RightForeArmEnd", "RightArmEnd", "LeftForeArmEnd", "LeftArmEnd", 
        "RightLegEnd", "RightUpLegEnd", "LeftLegEnd", "LeftUpLegEnd"
    ]

    remove_bones_idx = [
        4, 6, 
        33, 34, 59, 60, 
        66, 67, 73, 74
    ]

    new_bone_names = [
        'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'HeadTopEnd', 
        'RightShoulder', 'RightArm', 'RightForeArm', 
        'RightHand', 'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightHandThumb4', 
        'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightHandIndex4', 
        'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightHandMiddle4', 
        'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandRing4', 
        'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'RightHandPinky4', 
        'LeftShoulder', 'LeftArm', 'LeftForeArm', 
        'LeftHand', 'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', 'LeftHandThumb4', 
        'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', 'LeftHandIndex4', 
        'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandMiddle4', 
        'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftHandRing4', 
        'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3', 'LeftHandPinky4', 
        'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'RightToe_End', 
        'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LeftToe_End'
    ]

    new_parents = [
        -1, 0, 1, 2, 3, 4, 5,
        3, 7, 8, 9, 10, 11, 12, 13, 10, 15, 16, 17, 10, 19, 20, 21, 10, 23, 24, 25, 10, 27, 28, 29,
        3, 31, 32, 33, 34, 35, 36, 37, 34, 39, 40, 41, 34, 43, 44, 45, 34, 47, 48, 49, 34, 51, 52, 53,
        0, 55, 56, 57, 58,
        0, 60, 61, 62, 63
    ]

    # Remove unused bones and rearrange bone structure
    new_offsets = []
    new_positions = []
    new_rotations = []

    for i in range(0, 75):
        if i not in remove_bones_idx:
            new_offsets.append(bvh_dict['offsets'][i, :])
            new_positions.append(bvh_dict['positions'][:, i, :])
            new_rotations.append(bvh_dict['rotations'][:, i, :])

    new_offsets = np.array(new_offsets)
    new_positions = np.array(new_positions).transpose(1, 0, 2)
    new_rotations = np.array(new_rotations).transpose(1, 0, 2)

    # Define new BVH dict
    new_bvh_dict = bvh_dict
    new_bvh_dict['positions'] = new_positions
    new_bvh_dict['rotations'] = new_rotations
    new_bvh_dict['offsets'] = new_offsets
    new_bvh_dict['parents'] = new_parents
    new_bvh_dict['names'] = new_bone_names

    return new_bvh_dict
