import json
import os
import random
import sys
from argparse import ArgumentParser
from datetime import datetime

# import cv2
from textwrap import wrap

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests  # type: ignore
import torch
from base.exception import MApplicationException
from base.logger import MLogger
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(__file__, "../../mdm")))

import mdm.data_loaders.humanml.utils.paramUtil as paramUtil
from mdm.data_loaders.get_data import get_dataset_loader
from mdm.data_loaders.humanml.scripts.motion_process import recover_from_ric
from mdm.data_loaders.tensors import collate
from mdm.model.cfg_sampler import ClassifierFreeSampleModel
from mdm.utils import dist_util
from mdm.utils.fixseed import fixseed
from mdm.utils.model_util import create_model_and_diffusion, load_model_wo_clip
from mdm.utils.parser_util import add_base_options, add_data_options, add_diffusion_options, add_model_options, get_args_per_group_name

from parts.config import DirName

logger = MLogger(__name__)


def execute(args):
    logger.info("motion-diffusion-model 開始", decoration=MLogger.DECORATION_BOX)

    if not args.text:
        logger.error(
            "指定された指示文が空です。\n{text}",
            text=args.text,
            decoration=MLogger.DECORATION_BOX,
        )
        raise MApplicationException()

    try:
        argv = parse_argv()

        # 日付フォルダが指定されてない場合、生成する
        argv.output_dir = os.path.join(args.parent_dir, datetime.now().strftime("%Y%m%d_%H%M%S")) if "20" not in args.parent_dir else args.parent_dir
        os.makedirs(argv.output_dir, exist_ok=True)

        argv.text_prompt = translate(args.text, args.lang)
        argv.motion_length = args.seconds
        argv.num_repetitions = args.num_repetitions

        with open(os.path.join(argv.output_dir, "prompt.txt"), "w") as f:
            f.write(args.text)
            f.write("\n\n")
            f.write(translate(args.text, args.lang))
            f.write("\n\n")

        seed = argv.seed if argv.seed > 0 else random.randint(1, 1024)
        fixseed(seed)

        max_frames = 196
        fps = 30
        n_frames = min(max_frames, int(argv.motion_length * fps))
        dist_util.setup_dist(argv.device)

        print("Creating model and diffusion...")
        model, diffusion = create_model_and_diffusion(argv)

        print(f"Loading checkpoints from [{argv.model_path}]...")
        state_dict = torch.load(argv.model_path, map_location="cpu")
        load_model_wo_clip(model, state_dict)

        if argv.guidance_param != 1:
            model = ClassifierFreeSampleModel(model)  # wrapping model with the classifier-free sampler
        model.to(dist_util.dev())
        model.eval()  # disable random masking

        print("Loading dataset...")
        texts = [argv.text_prompt]
        argv.num_samples = 1

        argv.batch_size = argv.num_samples  # Sampling a single batch from the testset, with exactly argv.num_samples
        data = get_dataset_loader(
            name=argv.dataset, batch_size=argv.batch_size, num_frames=max_frames, datapath=argv.datapath, split="test", hml_mode="text_only"
        )
        data.fixed_length = n_frames

        _, model_kwargs = collate([{"inp": torch.tensor([[0.0]]), "target": 0, "text": txt, "tokens": None, "lengths": n_frames} for txt in texts])
        skeleton = paramUtil.t2m_kinematic_chain

        # 追跡画像用色生成
        cmap = plt.get_cmap("gist_rainbow")
        pid_colors = np.array([cmap(i) for i in np.linspace(0, 1, argv.num_repetitions)])
        idxs = np.arange(argv.num_repetitions)
        # 適当にばらけさせる
        cidxs = np.concatenate(
            [
                np.where(idxs % 5 == 0)[0],
                np.where(idxs % 5 == 1)[0][::-1],
                np.where(idxs % 5 == 2)[0],
                np.where(idxs % 5 == 3)[0][::-1],
                np.where(idxs % 5 == 4)[0],
            ]
        )
        pid_colors_opencv = [(np.array((c[2], c[1], c[0])) * 255).astype(int).tolist() for c in pid_colors[cidxs]]

        mdm_output_dir = os.path.join(argv.output_dir, DirName.MDM.value)
        os.makedirs(mdm_output_dir, exist_ok=True)

        for pidx in range(argv.num_repetitions):
            pname = f"{(pidx + 1):03d}"
            logger.info(
                "【No.{pname}】motion-diffusion-model 実行",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            personal_output_dir = os.path.join(argv.output_dir, DirName.PERSONAL.value, pname)
            os.makedirs(personal_output_dir, exist_ok=True)

            # add CFG scale to batch
            if argv.guidance_param != 1:
                model_kwargs["y"]["scale"] = torch.ones(argv.batch_size, device=dist_util.dev()) * argv.guidance_param

            sample_fn = diffusion.p_sample_loop

            sample = sample_fn(
                model,
                (argv.batch_size, model.njoints, model.nfeats, n_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )

            # Recover XYZ *positions* from HumanML3D vector representation
            if model.data_rep == "hml_vec":
                n_joints = 22 if sample.shape[1] == 263 else 21
                sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
                sample = recover_from_ric(sample, n_joints)
                sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

            motion = sample.cpu().numpy()[0].transpose(2, 0, 1) * 1.3
            json_data = {
                "color": [
                    float(pid_colors_opencv[pidx][2]) / 255,
                    float(pid_colors_opencv[pidx][1]) / 255,
                    float(pid_colors_opencv[pidx][0]) / 255,
                ],
                "estimation": motion.tolist(),
            }
            with open(os.path.join(mdm_output_dir, f"{pname}.json"), "w") as f:
                json.dump(json_data, f, indent=4)

            logger.info(
                "【No.{pname}】motion-diffusion-model 動画出力",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            plot_3d_motion(os.path.join(personal_output_dir, f"{pname}.avi"), skeleton, motion, title=argv.text_prompt, fps=fps)

            logger.info(
                "【No.{pname}】motion-diffusion-model 完了",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

        logger.info(
            "motion-diffusion-model 完了: {output_dir}",
            output_dir=argv.output_dir,
            decoration=MLogger.DECORATION_BOX,
        )

        return True, argv.output_dir
    except Exception as e:
        logger.critical("motion-diffusion-model で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        raise e


def plot_3d_motion(save_path, kinematic_tree, joints, title, figsize=(3, 3), fps=120, radius=3):
    matplotlib.use("Agg")

    title = "\n".join(wrap(title, 20))

    data = joints.copy().reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)
    plt.tight_layout()

    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = [
        "#DD5A37",
        "#D69E00",
        "#B75A39",
        "#DD5A37",
        "#D69E00",
        "#FF6D00",
        "#FF6D00",
        "#FF6D00",
        "#FF6D00",
        "#FF6D00",
        "#DDB50E",
        "#DDB50E",
        "#DDB50E",
        "#DDB50E",
        "#DDB50E",
    ]

    frame_number = data.shape[0]
    #     print(dataset.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    fourcc = cv2.VideoWriter_fourcc(*"I420")
    out = cv2.VideoWriter(
        save_path,
        fourcc,
        30.0,
        (300, 300),
    )

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3.0, radius * 2 / 3.0])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [[minx, miny, minz], [minx, miny, maxz], [maxx, miny, maxz], [maxx, miny, minz]]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    for index in tqdm(range(frame_number)):
        ax = Axes3D(fig)
        init()
        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1], MAXS[2] - trajec[index, 1])

        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth, color=color)

        plt.axis("off")

        # 一旦画像保存
        tmp_file_path = os.path.join(os.path.dirname(save_path), "tmp.jpg")
        plt.savefig(tmp_file_path)

        # 書き込み出力
        img = cv2.imread(tmp_file_path)
        out.write(img)

    out.release()
    cv2.destroyAllWindows()

    plt.close()


def translate(text: str, lang: str):
    if lang == "en":
        return text

    params = (
        ("text", text),
        ("source", lang),
        ("target", "en"),
    )

    # GASを叩く
    # https://qiita.com/satto_sann/items/be4177360a0bc3691fdf
    response = requests.get(
        "https://script.google.com/macros/s/AKfycbzZtvOvf14TaMdRIYzocRcf3mktzGgXvlFvyczo/exec",
        params=params,
    )

    # 結果を解析
    results = json.loads(response.text)

    if "text" in results:
        return results["text"]

    return text


def parse_argv():
    parser = ArgumentParser([])
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)

    group = parser.add_argument_group("sampling")
    group.add_argument(
        "--model_path",
        default="../data/motion-diffusion-model/humanml_trans_enc_512/model000200000.pt",
        type=str,
        help="Path to model####.pt file to be sampled.",
    )
    group.add_argument("--smpl_data_path", default="../data/motion-diffusion-model/smpl", type=str, help="Path to model####.pt file to be sampled.")
    group.add_argument(
        "--datapath", default="../data/motion-diffusion-model/humanml_opt.txt", type=str, help="Path to model####.pt file to be sampled."
    )
    group.add_argument(
        "--output_dir",
        default="",
        type=str,
        help="Path to results dir (auto created by the script). " "If empty, will create dir in parallel to checkpoint.",
    )
    group.add_argument(
        "--input_text", default="", type=str, help="Path to csv/txt file that specifies generation. If empty, will take text prompts from dataset."
    )
    group.add_argument("--text_prompt", default="", type=str, help="A text prompt to be generated. If empty, will take text prompts from dataset.")
    group.add_argument(
        "--num_samples",
        default=10,
        type=int,
        help="Maximal number of prompts to sample, " "if loading dataset from file, this field will be ignored.",
    )
    group.add_argument("--num_repetitions", default=3, type=int, help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument(
        "--motion_length",
        default=6.0,
        type=float,
        help="The length of the sampled motion [in seconds]. "
        "Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)",
    )
    group.add_argument(
        "--guidance_param", default=2.5, type=float, help="For classifier-free sampling - specifies the s parameter, as defined in the paper."
    )
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    argv = parser.parse_args(args=[])
    args_to_overwrite = []
    for group_name in ["dataset", "model", "diffusion"]:
        args_to_overwrite += get_args_per_group_name(parser, argv, group_name)

    # load args from model
    args_path = os.path.join(os.path.abspath(os.path.dirname(argv.model_path)), "args.json")
    assert os.path.exists(args_path), "Arguments json file was not found!"
    with open(args_path, "r") as fr:
        model_args = json.load(fr)
    for a in args_to_overwrite:
        if a in model_args.keys():
            argv.__dict__[a] = model_args[a]
        else:
            print("Warning: was not able to load [{}], using default value [{}] instead.".format(a, argv.__dict__[a]))

    return argv
