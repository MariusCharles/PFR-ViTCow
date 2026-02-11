from ..VideoMAE.kinetics import VideoClsDataset
import argparse
from ..config import NUM_FRAMES_PER_CLIP, FRAME_STEP, CROP_SIZE
def build_argparser() -> argparse.ArgumentParser:
    """
    Construit l'ArgumentParser pour VideoMAE (évaluation).

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="VideoMAE evaluation"
    )

    parser.add_argument(
        "--anno_path",
        type=str,
        required=True,
        help="Chemin vers le CSV ou fichier txt (path,label)"
    )

    parser.add_argument(
        "--nb_classes",
        type=int,
        required=True,
        help="Nombre total de classes"
    )

    parser.add_argument(
        "--test_num_segment",
        type=int,
        default=1,
        help="Nombre de segments temporels en test"
    )

    parser.add_argument(
        "--test_num_crop",
        type=int,
        default=1,
        help="Nombre de crops spatiaux en test"
    )

    return parser

if __name__ == "__main__":
    args = build_argparser().parse_args()
    dataset = VideoClsDataset(
            anno_path=args.anno_path,
            data_path='/',
            mode="test",
            clip_len=NUM_FRAMES_PER_CLIP,
            frame_sample_rate=FRAME_STEP, ##Fais à l'avance, donc peut-être 1 en fait
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=3,
            keep_aspect_ratio=True,
            crop_size=CROP_SIZE,
            short_side_size=CROP_SIZE,
            new_height=256,
            new_width=320,
            args=args)
