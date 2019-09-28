import argparse
from easydict import EasyDict as edict

from trainer.yolov3_train import trainer as yolov3_trainer
from trainer.fasterrcnn_train import train as fasterrcnn_trainer
from trainer.deepdsod_train import main as dsod_trainer
from trainer.deeplab_train import train as deeplab_trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default='../data', help="path to image folder")
    parser.add_argument("--model", type=str, help="choose model(fasterrcnn, deeplabv3, yolov3 or deepdsod) to train")
    opt = parser.parse_args()
    print(opt)
    
    if opt.model not in ['yolov3', 'fasterrcnn', 'deeplabv3', 'deepdsod']:
        print('Must choose a model to train!(fasterrcnn, deeplabv3, yolov3 or deepdsod)')

    elif opt.model == 'yolov3':
          hyperparameters = {
            'root': opt.image_folder,
            'epochs': 200,
            'batch_size': 8,
            'gradient_accumulations': 2, 
            'model_def': 'yolov3/config/yolov3-custom.cfg',
            'data_config': 'yolov3/config/custom.data',
            'pretrained_weights': None,
            'n_cpu': 4,
            'img_size': 608,
            'checkpoint_interval': 1,
            'evaluation_interval': 1,
            'compute_map': False,
            'multiscale_training': True,
           }
          yolov3_trainer(edict(hyperparameters))
          
    elif opt.model == 'fasterrcnn':
          fasterrcnn_trainer(epochs=100, print_freq=200)
          
    elif opt.model == 'deeplabv3':
          hyperparameters = {
            'root': opt.image_folder, 
            'lr': 1e-2,
            'epochs': 100,
            'decay_step': 100*0.1,
            'momentum': 0.9,
            'weight_decay': 1e-5,
            'batch_size': 2,
            'set_BN_momentum': False,
            'AccumulateStep': 2,
           }
          deeplab_trainer(edict(hyperparameters))
          
    elif opt.model == 'deepdsod':
          hyperparameters = {
                'root': opt.image_folder,
                'checkpoint': None,
                'lr_scheduler': None,
                'batch_size': 4,
                'start_epoch': 0,
                'epochs': 200,
                'epochs_since_improvement': 0,
                'best_loss': 100,
                'workers': 4,
                'print_freq': 200,
                'lr': 1e-3,
                'momentum': 0.9,
                'weight_decay': 5e-4,
                'grad_clip': None,
           }
          dsod_trainer(edict(hyperparameters))
