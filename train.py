import os
import sys
#stderr = sys.stderr
#sys.stderr = open(os.devnull, 'w')
sys.path.append('libs/')  
import gc
import numpy as np
import matplotlib.pyplot as plt
# Import backend without the "Using X Backend" message
from argparse import ArgumentParser
from PIL import Image
from libs.cisrdcnn import CISRDCNN
from libs.util import plot_test_images, DataLoader
from tensorflow.keras import backend as K


# Sample call
"""
# Train 2X CISRDCNN
python3 train.py --train ../../Documents/data/train_large/ --validation ../data/val_large/ --test ../data/benchmarks/Set5/  --log_test_path ./test/ --scale 2 --stage dbcnn

# Train the 4X CISRDCNN
python3 train.py --train ../../data/train_large/ --validation ../data/val_large/ --test ../data/benchmarks/Set5/  --log_test_path ./test/ --scale 4 --scaleFrom 2 --stage all

# Train the 8X CISRDCNN
python3 train.py --train ../../data/train_large/ --validation ../data/val_large/ --test ../data/benchmarks/Set5/  --log_test_path ./test/ --scale 8 --scaleFrom 4 --stage all
"""

def parse_args():
    parser = ArgumentParser(description='Training script for CISRDCNN')

    parser.add_argument(
        '-stage', '--stage',
        type=str, default='all',
        help='Which stage of training to run',
        choices=['all', 'dbcnn', 'uscnn', 'qecnn','cisrdcnn']
    )

    parser.add_argument(
        '-epochs', '--epochs',
        type=int, default=30,
        help='Number epochs per train'
    )

    parser.add_argument(
        '-train', '--train',
        type=str, default='../../Documents/data/train_large/',
        help='Folder with training images'
    )

    parser.add_argument(
        '-steps_per_epoch', '--steps_per_epoch',
        type=int, default=256,
        help='Steps per epoch'
    )

    parser.add_argument(
        '-validation', '--validation',
        type=str, default='../data/val_large/',
        help='Folder with validation images'
    )

    parser.add_argument(
        '-steps_per_validation', '--steps_per_validation',
        type=int, default=16,
        help='Steps per validation'
    )
    
    parser.add_argument(
        '-test', '--test',
        type=str, default='../data/benchmarks/Set5/',
        help='Folder with testing images'
    )

    parser.add_argument(
        '-print_frequency', '--print_frequency',
        type=int, default=15,
        help='Frequency of print test images'
    )
        
    parser.add_argument(
        '-modelname', '--modelname',
        type=str, default='CISRDCNN',
        help='CISRDCNN'
    )
        
    parser.add_argument(
        '-scale', '--scale',
        type=int, default=2,
        help='How much should we upscale images'
    )

    parser.add_argument(
        '-scaleFrom', '--scaleFrom',
        type=int, default=None,
        help='Perform transfer learning from lower-upscale model'
    )
        
    parser.add_argument(
        '-workers', '--workers',
        type=int, default=4,
        help='How many workers to user for pre-processing'
    )

    parser.add_argument(
        '-max_queue_size', '--max_queue_size',
        type=int, default=10,
        help='Max queue size to workers'
    )
        
    parser.add_argument(
        '-batch_size', '--batch_size',
        type=int, default=32,
        help='What batch-size should we use'
    )

    parser.add_argument(
        '-crops_per_image', '--crops_per_image',
        type=int, default=16,
        help='Increase in order to reduce random reads on disk (in case of slower SDDs or HDDs)'
    )           
        
    parser.add_argument(
        '-weight_path', '--weight_path',
        type=str, default='./model/',
        help='Where to output weights during training'
    )

    parser.add_argument(
        '-log_tensorboard_update_freq', '--log_tensorboard_update_freq',
        type=int, default=10,
        help='Frequency of update tensorboard weight'
    )
        
    parser.add_argument(
        '-log_path', '--log_path',
        type=str, default='./logs/',
        help='Where to output tensorboard logs during training'
    )

    parser.add_argument(
        '-log_test_path', '--log_test_path',
        type=str, default='./test/',
        help='Path to generate images in train'
    )


    parser.add_argument(
        '-height_lr', '--height_lr',
        type=int, default=64,
        help='height of lr crop'
    )

    parser.add_argument(
        '-width_lr', '--width_lr',
        type=int, default=64,
        help='width of lr crop'
    )

    parser.add_argument(
        '-channels', '--channels',
        type=int, default=3,
        help='channels of images'
    )

    parser.add_argument(
        '-colorspace', '--colorspace',
        type=str, default='RGB',
        help='Colorspace of images, e.g., RGB or YYCbCr'
    )

    parser.add_argument(
        '-media_type', '--media_type',
        type=str, default='i',
        help='Type of media i to image or v to video'
    )

    parser.add_argument(
        '-qf', '--qf',
        type=int, default=20,
        help='parameter of level quantization'
    )
        
    return  parser.parse_args()



# Run script
def main():

    # Parse command-line arguments
    args = parse_args()
       
    # Common settings for all training stages
    args_train = { 
        "batch_size": args.batch_size, 
        "steps_per_epoch": args.steps_per_epoch,
        "steps_per_validation": args.steps_per_validation,
        "crops_per_image": args.crops_per_image,
        "print_frequency": args.print_frequency,
        "log_tensorboard_update_freq": args.log_tensorboard_update_freq,
        "workers": args.workers,
        "max_queue_size": args.max_queue_size,
        "datapath_train": args.train,
        "datapath_validation": args.validation,
        "datapath_test": args.test,
        "log_weight_path": args.weight_path, 
        "log_tensorboard_path": args.log_path,        
        "log_test_path": args.log_test_path,        
        "media_type": args.media_type,
        "qf": args.qf
    }

    args_model = {
        "height_lr": args.height_lr, 
        "width_lr": args.width_lr, 
        "channels": args.channels, 
        "colorspace": args.colorspace,
        "stage": args.stage,
        "upscaling_factor": args.scale        
    }

    # Weight paths
    dbcnn_path = os.path.join(args.weight_path, 'DBCNN_'+str(args.scale)+'X.tf')
    uscnn_path = os.path.join(args.weight_path, 'USCNN_'+str(args.scale)+'X.tf')
    qecnn_path = os.path.join(args.weight_path, 'QECNN_'+str(args.scale)+'X.tf')
    cisrdcnn_path = os.path.join(args.weight_path, args.modelname+'_'+str(args.scale)+'X.tf')
    

    ## FIRST STAGE: Train the deblocking network DBCNN
    ######################################################
    
    if args.stage in ['all', 'dbcnn']:
        print(">> FIRST STAGE: Train the deblocking network DBCNN: scale {}X".format(args.scale))
        args_model['stage'] = 'dbcnn'
        if args.scaleFrom:
            print(">> Train the deblocking network DBCNN: scale {}X with transfer learning from {}X".format(args.scale,args.scaleFrom))
            #TODO: transferência de aprendizagem para escala maior    
        else:
            cisrdcnn = CISRDCNN(lr=1e-1, **args_model) 
            cisrdcnn.train_dbcnn(epochs=args.epochs,model_name="DBCNN",**args_train)

    ## SECOND STAGE: Train the upsampling network USCNN
    ######################################################  
    if args.stage in ['all', 'uscnn']:
        print("SECOND STAGE: Train the upsampling network USCNN: scale {}X".format(args.scale))
        args_model['stage'] = 'uscnn'
        cisrdcnn = CISRDCNN(lr=1e-3, **args_model)
        print("Loading weights DBCNN...")
        cisrdcnn.dbcnn.load_weights(dbcnn_path)
        cisrdcnn.train_uscnn(epochs=args.epochs,model_name="USCNN",**args_train)
    
    ## THIRD STAGE: Train the quality enhancement network QECNN
    ######################################################  
    if args.stage in ['all', 'qecnn']:
        print("THIRD STAGE: Train the quality enhancement network QECNN: scale {}X".format(args.scale))
        args_model['stage'] = 'qecnn'
        cisrdcnn = CISRDCNN(lr=1e-2, **args_model)
        print("Loading weights DBCNN")
        cisrdcnn.dbcnn.load_weights(dbcnn_path)
        print("Loading weights USCNN")
        cisrdcnn.uscnn.load_weights(uscnn_path)
        cisrdcnn.train_qecnn(epochs=args.epochs,model_name="QECNN",**args_train)
    
    ## FOURTH STAGE: the CISRDCNN is optimized in an end-to-end manner
    ######################################################  
    if args.stage in ['all', 'cisrdcnn']:
        print("FOURTH STAGE: the CISRDCNN is optimized in an end-to-end manner: scale {}X".format(args.scale))
        args_model['stage'] = 'cisrdcnn'
        cisrdcnn = CISRDCNN(lr=1e-4, **args_model)
        print("Loading weights DBCNN")
        cisrdcnn.dbcnn.load_weights(dbcnn_path)
        print("Loading weights USCNN")
        cisrdcnn.uscnn.load_weights(uscnn_path)
        print("Loading weights QECNN")
        cisrdcnn.qecnn.load_weights(qecnn_path)
        if args.stage in ['cisrdcnn']:  
            cisrdcnn.cisrdcnn.load_weights(cisrdcnn_path)
        print(cisrdcnn.cisrdcnn.summary())
        cisrdcnn.train_cisrdcnn(epochs=args.epochs,model_name="CISRDCNN",**args_train)



# Run the CISRDCNN network
if __name__ == "__main__":
    main()
        
