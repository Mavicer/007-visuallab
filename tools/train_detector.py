"""One bounded local detector experiment. OCR characters are not YOLO classes."""
import argparse
import json
from pathlib import Path
from time import perf_counter


def main() -> None:
    parser = argparse.ArgumentParser(description='本地 YOLO 定位试验；先审查标注与数据划分')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=Path('models/yolo26n.pt'))
    parser.add_argument('--output', type=Path, default=Path('runs/card-pilot'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--image-size', type=int, default=416,
                        help='固定训练尺寸；小文字试验使用 640，纸片旧基线为 416')
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--profile', choices=('baseline', 'tilt', 'armor'), default='baseline',
                        help='tilt 为文字姿态微调；armor 为整板定位；类别以数据 YAML 为准，均不训练 OCR')
    parser.add_argument('--dry-run', action='store_true', help='只打印完整配置，不加载模型或训练')
    args = parser.parse_args()
    if not args.data.is_file() or not args.weights.is_file():
        parser.error('请提供已有的本地数据 YAML 与预训练权重')
    if not 1 <= args.epochs <= 20:
        parser.error('单次试验 epochs 必须为 1～20；超出预算需另行制定计划')
    if args.image_size < 32 or args.image_size % 32:
        parser.error('image-size 必须是至少 32 的 32 倍数')
    train_args = dict(
        data=str(args.data.resolve()), device=args.device,
        epochs=args.epochs, patience=5, imgsz=args.image_size, batch=4, nbs=4, workers=0,
        freeze=10, optimizer='AdamW', lr0=.001, amp=False,
        fliplr=0.0, degrees=10, close_mosaic=5,
        seed=42, deterministic=True, plots=False, verbose=False,
        project=str(args.output.resolve().parent), name=args.output.name,
        exist_ok=False,
    )
    if args.profile == 'tilt':
        # Fine-tune the existing text head with conservative geometry changes.
        # No mirrors: ASCII glyph semantics should not be inverted. Disable
        # mosaic to avoid shrinking already small characters further.
        train_args.update(lr0=.0005, degrees=60., perspective=.0004, shear=5.,
                          translate=.08, scale=.3, mosaic=0., close_mosaic=0,
                          flipud=0., fliplr=0., hsv_h=.015, hsv_s=.4, hsv_v=.25)
    if args.profile == 'armor':
        # Whole-marker localization is a separate task from ASCII reading.
        # Real blurred positives are retained; augmentation is deliberately
        # bounded so the small marker is not reduced by mosaic composition.
        train_args.update(lr0=.001, degrees=60., perspective=.0005, shear=5.,
                          translate=.1, scale=.35, mosaic=0., close_mosaic=0,
                          flipud=0., fliplr=0., hsv_h=.02, hsv_s=.5, hsv_v=.35)
    if args.dry_run:
        print(json.dumps({'weights':str(args.weights.resolve()),'profile':args.profile,
                          'train_args':train_args},ensure_ascii=False,indent=2))
        return
    if args.output.exists():
        parser.error('output 已存在；为避免意外重复训练，请使用新目录并审查训练预算')
    import torch
    from ultralytics import YOLO, settings
    settings.update({'sync': False})
    torch.set_num_threads(4)
    started = perf_counter()
    model = YOLO(str(args.weights.resolve()))
    model.train(**train_args)
    summary = {'elapsed_seconds': round(perf_counter()-started, 1),
               'best_weights': str(model.trainer.best),
               'data': str(args.data.resolve()), 'requested_epochs': args.epochs,
        'device': args.device, 'image_size': args.image_size,
        'profile':args.profile,'initial_weights':str(args.weights.resolve()),
        'training_parameters':train_args,
        'note': 'Small local experiment, not a deployment accuracy guarantee.'}
    (model.trainer.save_dir/'experiment.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
