import argparse
import os
import sys
import time

from base.logger import MLogger

logger = MLogger(__name__)


def show_worked_time(elapsed_time):
    # 経過秒数を時分秒に変換
    td_m, td_s = divmod(elapsed_time, 60)
    td_h, td_m = divmod(td_m, 60)

    if td_m == 0:
        worked_time = "00:00:{0:02d}".format(int(td_s))
    elif td_h == 0:
        worked_time = "00:{0:02d}:{1:02d}".format(int(td_m), int(td_s))
    else:
        worked_time = "{0:02d}:{1:02d}:{2:02d}".format(int(td_h), int(td_m), int(td_s))

    return worked_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, dest="text", default="", help="text (English or Japanese)")
    parser.add_argument("--seconds", type=float, dest="seconds", default=5.0, help="seconds")
    parser.add_argument("--num_repetitions", type=int, dest="num_repetitions", default=3, help="num_repetitions")
    parser.add_argument("--seed", type=int, dest="seed", default=3, help="seed")
    parser.add_argument(
        "--parent-dir",
        type=str,
        dest="parent_dir",
        default="",
        help="Process parent dir path",
    )
    parser.add_argument("--process", type=str, dest="process", default="", help="Process to be executed")
    parser.add_argument(
        "--trace-check-model-config",
        type=str,
        dest="trace_check_model_config",
        default=os.path.abspath(os.path.join(__file__, "../../data/pmx/trace_check_model.pmx")),
        help="MMD Model Bone pmx",
    )
    parser.add_argument(
        "--trace-rot-model-config",
        type=str,
        dest="trace_rot_model_config",
        default=os.path.abspath(os.path.join(__file__, "../../data/pmx/trace_model.pmx")),
        help="MMD Model Bone pmx",
    )
    parser.add_argument("--verbose", type=int, dest="verbose", default=20, help="Log level")
    parser.add_argument("--log-mode", type=int, dest="log_mode", default=0, help="Log output mode")
    parser.add_argument("--lang", type=str, dest="lang", default="en", help="Language")

    args = parser.parse_args()
    MLogger.initialize(level=args.verbose, mode=args.log_mode, lang=args.lang)
    result = True

    start = time.time()

    logger.info(
        "MMDモーション自動生成ツール\n　テキスト: {text}\n　処理内容: {process}",
        text=args.text,
        process=args.process,
        decoration=MLogger.DECORATION_BOX,
    )

    try:
        if "text2move" in args.process:
            # 準備
            from parts.text2move import execute

            result, args.output_dir = execute(args)

        if result and "mix" in args.process:
            # 推定結果合成
            from parts.mix import execute

            result = execute(args)

        if result and "motion" in args.process:
            # モーション生成
            from parts.motion import execute

            result = execute(args)

        elapsed_time = time.time() - start

        logger.info(
            "MMDモーション自動生成ツール終了\n　テキスト: {text}\n　処理内容: {process}\n　生成結果: {output_dir}\n　処理時間: {elapsed_time}",
            text=args.text,
            process=args.process,
            output_dir=args.output_dir,
            elapsed_time=show_worked_time(elapsed_time),
            decoration=MLogger.DECORATION_BOX,
        )
    except Exception as e:
        elapsed_time = time.time() - start

        import traceback

        print(traceback.format_exc())

        logger.error(
            "MMDモーション自動生成ツール失敗\n　テキスト: {text}\n　処理内容: {process}\n　処理時間: {elapsed_time}",
            text=args.text,
            process=args.process,
            elapsed_time=show_worked_time(elapsed_time),
            decoration=MLogger.DECORATION_BOX,
        )
        # 例外が発生したら終了ログ出力
        logger.quit()
    finally:
        # 終了音を鳴らす
        if os.name == "nt":
            # Windows
            try:
                import winsound

                winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS)
            except Exception:
                pass
