import json
import os
import re
from datetime import datetime
from glob import glob

import numpy as np
from base.exception import MApplicationException
from base.logger import MLogger
from base.math import MMatrix4x4, MQuaternion, MVector3D
from tqdm import tqdm

from parts.config import SMPL_JOINT_22, DirName

logger = MLogger(__name__)

# 身長158cmプラグインより
MIKU_CM = 0.1259496


def execute(args):
    logger.info(
        "推定結果合成 処理開始: {output_dir}",
        output_dir=args.output_dir,
        decoration=MLogger.DECORATION_BOX,
    )

    if not os.path.exists(args.output_dir):
        logger.error(
            "指定された処理用ディレクトリが存在しません。: {output_dir}",
            output_dir=args.output_dir,
            decoration=MLogger.DECORATION_BOX,
        )
        raise MApplicationException()

    try:
        for personal_json_path in sorted(glob(os.path.join(args.output_dir, DirName.MDM.value, "*.json"))):
            pname, _ = os.path.splitext(os.path.basename(personal_json_path))

            logger.info(
                "【No.{pname}】推定結果合成 合成開始",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            frame_joints = {}
            with open(personal_json_path, "r") as f:
                frame_joints = json.load(f)

            initial_nose_pos = MVector3D(0, 17.59783, -1.127905)
            initial_head_pos = MVector3D(0, 17.33944, 0.3088881)
            initial_neck_pos = MVector3D(0, 16.42476, 0.4232453)
            initial_direction: MVector3D = (initial_nose_pos - initial_head_pos).normalized()
            initial_up: MVector3D = (initial_head_pos - initial_neck_pos).normalized()
            initial_cross: MVector3D = initial_up.cross(initial_direction).normalized()
            initial_head_qq = MQuaternion.from_direction(initial_direction, initial_cross)

            initial_left_ear_pos = (MVector3D(1.147481, 17.91739, 0.4137991) - initial_nose_pos) / MIKU_CM
            initial_right_ear_pos = (MVector3D(-1.147481, 17.91739, 0.4137991) - initial_nose_pos) / MIKU_CM

            mix_joints = {"color": frame_joints["color"], "joints": {}}
            for fno in tqdm(range(len(frame_joints["estimation"])), desc=f"No.{pname} ... "):
                mix_joints["joints"][fno] = {}
                for jname, (xval, yval, zval) in zip(SMPL_JOINT_22.keys(), frame_joints["estimation"][fno]):
                    mix_joints["joints"][fno][jname] = {
                        "x": xval * 100,
                        "y": yval * 100,
                        "z": -zval * 100,
                    }

                mix_joints["joints"][fno]["Pelvis2"] = {}
                mix_joints["joints"][fno]["Neck"] = {}

                for axis in ("x", "y", "z"):
                    # 下半身先
                    mix_joints["joints"][fno]["Pelvis2"][axis] = np.mean(
                        [
                            mix_joints["joints"][fno]["LHip"][axis],
                            mix_joints["joints"][fno]["RHip"][axis],
                        ]
                    )

                    # 首
                    mix_joints["joints"][fno]["Neck"][axis] = np.mean(
                        [
                            mix_joints["joints"][fno]["Head"][axis],
                            mix_joints["joints"][fno]["Spine3"][axis],
                        ]
                    )

                # 耳位置を暫定で求める
                head_pos = MVector3D(
                    mix_joints["joints"][fno]["Head"]["x"],
                    mix_joints["joints"][fno]["Head"]["y"],
                    mix_joints["joints"][fno]["Head"]["z"],
                )
                neck_pos = MVector3D(
                    mix_joints["joints"][fno]["Neck"]["x"],
                    mix_joints["joints"][fno]["Neck"]["y"],
                    mix_joints["joints"][fno]["Neck"]["z"],
                )
                nose_pos = MVector3D(
                    mix_joints["joints"][fno]["Nose"]["x"],
                    mix_joints["joints"][fno]["Nose"]["y"],
                    mix_joints["joints"][fno]["Nose"]["z"],
                )

                direction: MVector3D = (nose_pos - head_pos).normalized()
                up: MVector3D = (head_pos - neck_pos).normalized()
                cross: MVector3D = up.cross(direction).normalized()
                head_qq = MQuaternion.from_direction(direction, cross) * initial_head_qq.inverse()

                ear_mat = MMatrix4x4(identity=True)
                ear_mat.translate(nose_pos)
                ear_mat.rotate(head_qq)

                left_ear_pos = ear_mat * initial_left_ear_pos
                right_ear_pos = ear_mat * initial_right_ear_pos

                mix_joints["joints"][fno]["LEar"] = {
                    "x": float(left_ear_pos.x),
                    "y": float(left_ear_pos.y),
                    "z": float(left_ear_pos.z),
                }
                mix_joints["joints"][fno]["REar"] = {
                    "x": float(right_ear_pos.x),
                    "y": float(right_ear_pos.y),
                    "z": float(right_ear_pos.z),
                }

            logger.info(
                "【No.{pname}】推定結果合成 出力開始",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            with open(
                os.path.join(args.output_dir, DirName.PERSONAL.value, pname, os.path.basename(personal_json_path)),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(mix_joints, f, indent=4)

        logger.info(
            "推定結果合成 処理終了: {output_dir}",
            output_dir=args.output_dir,
            decoration=MLogger.DECORATION_BOX,
        )

        return True
    except Exception as e:
        logger.critical("推定結果合成で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        raise e
