import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ReLU, BatchNormalization, Add 
from tensorflow.keras.layers import UpSampling2D,Conv2DTranspose

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.initializers import RandomNormal


import restore 
from util import DataLoader, plot_test_images
from losses import psnr3 as psnr
from losses import euclidean, cosine, charbonnier

class CISRDCNN():
    
    def __init__(self,
                 height_lr=16, width_lr=16, channels=3,
                 upscaling_factor=4, lr = 1e-4,
                 training_mode=True,
                 colorspace = 'RGB'
                 ):

        # Low-resolution image dimensions
        self.height_lr = height_lr
        self.width_lr = width_lr

        # High-resolution image dimensions
        if upscaling_factor not in [1, 2, 4, 8]:
            raise ValueError(
                'Upscaling factor must be either 2, 4, or 8. You chose {}'.format(upscaling_factor))
        self.upscaling_factor = upscaling_factor
        self.height_hr = int(self.height_lr * self.upscaling_factor)
        self.width_hr = int(self.width_lr * self.upscaling_factor)

        # Low-resolution and high-resolution shapes
        self.channels = channels
        self.colorspace = colorspace

        self.shape_lr = (self.height_lr, self.width_lr, self.channels)
        self.shape_hr = (self.height_hr, self.width_hr, self.channels)

        self.loss = "mse"
        self.lr = lr

        self.dbcnn = self.build_dbcnn()
        self.compile_dbcnn(self.dbcnn)

        self.uscnn = self.build_uscnn()
        self.compile_uscnn(self.uscnn)


    def save_weights(self, filepath):
        """Save the networks weights"""
        self.dbcnn.save_weights(
            "{}_{}X.h5".format(filepath, self.upscaling_factor))
        

    def load_weights(self, weights=None, **kwargs):
        """ Load the network weights """
        print(">> Loading weights...")
        if weights:
            self.dbcnn.load_weights(weights, **kwargs)
        
    
    def compile_dbcnn(self, model):
        """Compile the DBCNN with appropriate optimizer"""
        
        model.compile(
            loss=self.loss,
            optimizer= SGD(lr=self.lr, momentum=0.9, decay=1e-6, nesterov=True), #Adam(lr=self.lr,beta_1=0.9, beta_2=0.999), 
            metrics=[psnr]
        )
    
    def compile_uscnn(self, model):
        """Compile the DBCNN with appropriate optimizer"""
        
        model.compile(
            loss=self.loss,
            optimizer= SGD(lr=self.lr, momentum=0.9, decay=1e-6, nesterov=True), #Adam(lr=self.lr,beta_1=0.9, beta_2=0.999), 
            metrics=[psnr]
        )

    def build_dbcnn(self,k1=20):

        
        def DBCNN(input):
            x=input
            for i in range(k1-1):
                x = Conv2D(filters= 64, kernel_size = (3,3), strides=1,padding='same')(x)
                x =  BatchNormalization()(x)
                x = ReLU()(x)
            x = Conv2D(filters= self.channels, kernel_size = (3,3), strides=1, padding='same', name='K1')(x)
            x = ReLU()(x)
            x = Add()([x, input])
            return x

        inputs = Input(shape=(None, None, self.channels))
        x = DBCNN(inputs)


        model = Model(inputs=inputs, outputs=x)
        logging.debug(model.summary())
        return model
    
    def build_uscnn(self,k2=10):
        def USCNN(x):
            for i in range(k2-1):
                x = Conv2D(filters= 64, kernel_size = (3,3), strides=1,padding='same')(x)
                x =  BatchNormalization()(x)
                x = ReLU()(x)
            x = UpSampling2D(size=(self.upscaling_factor, self.upscaling_factor),interpolation="nearest")(x)
            x = Conv2D(filters= self.channels, kernel_size = (9,9), strides=1,padding='same')(x)
            #x = Conv2DTranspose(filters= self.channels, kernel_size = (9,9), strides=1, padding='valid', name='K2')(x)
            x = ReLU()(x)
            return x
        inputs = Input(shape=(16, 16, self.channels))
        x = USCNN(inputs)

        model = Model(inputs=inputs, outputs=x)
        logging.debug(model.summary())
        return model

    def train_dbcnn(self,
            epochs=50,
            batch_size=8,
            steps_per_epoch=5,
            steps_per_validation=5,
            crops_per_image=4,
            print_frequency=5,
            log_tensorboard_update_freq=10,
            workers=1,
            max_queue_size=5,
            model_name='CISRDCNN',
            media_type='i', 
            datapath_train='../../../videos_harmonic/MYANMAR_2160p/train/',
            datapath_validation='../../../videos_harmonic/MYANMAR_2160p/validation/',
            datapath_test='../../../videos_harmonic/MYANMAR_2160p/test/',
            log_weight_path='../model/', 
            log_tensorboard_path='../logs/',
            log_test_path='../test/'
        ):

        # Create data loaders
        
        train_loader = DataLoader(
            datapath_train, batch_size,
            self.height_hr, self.width_hr,
            self.upscaling_factor,
            crops_per_image,
            media_type,
            self.channels,
            self.colorspace
        )
        

        validation_loader = None 
        if datapath_validation is not None:
            validation_loader = DataLoader(
                datapath_validation, batch_size,
                self.height_hr, self.width_hr,
                self.upscaling_factor,
                crops_per_image,
                media_type,
                self.channels,
                self.colorspace
        )

        test_loader = None
        if datapath_test is not None:
            test_loader = DataLoader(
                datapath_test, 1,
                self.height_hr, self.width_hr,
                self.upscaling_factor,
                1,
                media_type,
                self.channels,
                self.colorspace
        )

        # Callback: tensorboard
        callbacks = []
        if log_tensorboard_path:
            tensorboard = TensorBoard(
                log_dir=os.path.join(log_tensorboard_path, model_name),
                histogram_freq=0,
                write_graph=True,
                update_freq=log_tensorboard_update_freq
            )
            callbacks.append(tensorboard)
        else:
            print(">> Not logging to tensorboard since no log_tensorboard_path is set")

        # Callback: Stop training when a monitored quantity has stopped improving
        earlystopping = EarlyStopping(
            monitor='val_loss', 
            patience=30, verbose=1, 
            restore_best_weights=True )
        callbacks.append(earlystopping)

        # Callback: Reduce lr when a monitored quantity has stopped improving
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=5, min_lr=1e-6,verbose=1)
        #callbacks.append(reduce_lr)

        # Callback: save weights after each epoch
        modelcheckpoint = ModelCheckpoint(
            os.path.join(log_weight_path, model_name + '_{}X.h5'.format(self.upscaling_factor)), 
            monitor='val_loss', 
            save_best_only=True, 
            save_weights_only=True)
        callbacks.append(modelcheckpoint)
  
        # Callback: test images plotting
        if datapath_test is not None:
            testplotting = LambdaCallback(
                on_epoch_end=lambda epoch, logs: None if ((epoch+1) % print_frequency != 0 ) else plot_test_images(
                    self.dbcnn,
                    test_loader,
                    datapath_test,
                    log_test_path,
                    epoch+1,
                    name=model_name,
                    channels=self.channels,
                    colorspace=self.colorspace))
        callbacks.append(testplotting)

        #callbacks.append(TQDMCallback())

        self.dbcnn.fit(
            train_loader,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_loader,
            validation_steps=steps_per_validation,
            callbacks=callbacks,
            shuffle=True,
            use_multiprocessing=False,
            workers=workers
        )
    
    def train_uscnn(self,
            epochs=50,
            batch_size=8,
            steps_per_epoch=5,
            steps_per_validation=5,
            crops_per_image=4,
            print_frequency=5,
            log_tensorboard_update_freq=10,
            workers=1,
            max_queue_size=5,
            model_name='CISRDCNN',
            media_type='i', 
            datapath_train='../../../videos_harmonic/MYANMAR_2160p/train/',
            datapath_validation='../../../videos_harmonic/MYANMAR_2160p/validation/',
            datapath_test='../../../videos_harmonic/MYANMAR_2160p/test/',
            log_weight_path='../model/', 
            log_tensorboard_path='../logs/',
            log_test_path='../test/'
        ):

        # Create data loaders
        
        train_loader = DataLoader(
            datapath_train, batch_size,
            self.height_hr, self.width_hr,
            self.upscaling_factor,
            crops_per_image,
            media_type,
            self.channels,
            self.colorspace
        )
        

        validation_loader = None 
        if datapath_validation is not None:
            validation_loader = DataLoader(
                datapath_validation, batch_size,
                self.height_hr, self.width_hr,
                self.upscaling_factor,
                crops_per_image,
                media_type,
                self.channels,
                self.colorspace
        )

        test_loader = None
        if datapath_test is not None:
            test_loader = DataLoader(
                datapath_test, 1,
                self.height_hr, self.width_hr,
                self.upscaling_factor,
                1,
                media_type,
                self.channels,
                self.colorspace
        )

        # Callback: tensorboard
        callbacks = []
        if log_tensorboard_path:
            tensorboard = TensorBoard(
                log_dir=os.path.join(log_tensorboard_path, model_name),
                histogram_freq=0,
                write_graph=True,
                update_freq=log_tensorboard_update_freq
            )
            callbacks.append(tensorboard)
        else:
            print(">> Not logging to tensorboard since no log_tensorboard_path is set")

        # Callback: Stop training when a monitored quantity has stopped improving
        earlystopping = EarlyStopping(
            monitor='val_loss', 
            patience=30, verbose=1, 
            restore_best_weights=True )
        callbacks.append(earlystopping)

        # Callback: Reduce lr when a monitored quantity has stopped improving
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=5, min_lr=1e-6,verbose=1)
        #callbacks.append(reduce_lr)

        # Callback: save weights after each epoch
        modelcheckpoint = ModelCheckpoint(
            os.path.join(log_weight_path, model_name + '_{}X.h5'.format(self.upscaling_factor)), 
            monitor='val_loss', 
            save_best_only=True, 
            save_weights_only=True)
        callbacks.append(modelcheckpoint)
  
        # Callback: test images plotting
        if datapath_test is not None:
            testplotting = LambdaCallback(
                on_epoch_end=lambda epoch, logs: None if ((epoch+1) % print_frequency != 0 ) else plot_test_images(
                    self.uscnn,
                    test_loader,
                    datapath_test,
                    log_test_path,
                    epoch+1,
                    name=model_name,
                    channels=self.channels,
                    colorspace=self.colorspace))
        callbacks.append(testplotting)

        #callbacks.append(TQDMCallback())

        self.uscnn.fit(
            train_loader,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_loader,
            validation_steps=steps_per_validation,
            callbacks=callbacks,
            shuffle=True,
            use_multiprocessing=False,
            workers=workers
        )


    def predict_dbcnn(self,
            lr_path = None,
            sr_path = None,
            print_frequency = False,
            qp = 8,
            fps = None,
            media_type = None,
            gpu = False
        ):
        """ lr_videopath: path of video in low resoluiton
            sr_videopath: path to output video 
            print_frequency: print frequncy the time per frame and estimated time, if False no print 
            crf: [0,51] QP parameter 0 is the best quality and 51 is the worst one
            fps: framerate if None is use the same framerate of the LR video
            media_type: type of media 'v' to video and 'i' to image
        """
        if(media_type == 'v'):
            time_elapsed = restore.write_srvideo(self.dbcnn,lr_path,sr_path,self.upscaling_factor,print_frequency=print_frequency,crf=qp,fps=fps,gpu=gpu)
        elif(media_type == 'i'):
            time_elapsed = restore.write_sr_images(self.dbcnn, lr_imagepath=lr_path, sr_imagepath=sr_path,scale=self.upscaling_factor)
        else:
            print(">> Media type not defined or not suported!")
            return 0
        return time_elapsed
    

    def predict_uscnn(self,
            lr_path = None,
            sr_path = None,
            print_frequency = False,
            qp = 8,
            fps = None,
            media_type = None,
            gpu = False
        ):
        """ lr_videopath: path of video in low resoluiton
            sr_videopath: path to output video 
            print_frequency: print frequncy the time per frame and estimated time, if False no print 
            crf: [0,51] QP parameter 0 is the best quality and 51 is the worst one
            fps: framerate if None is use the same framerate of the LR video
            media_type: type of media 'v' to video and 'i' to image
        """
        if(media_type == 'v'):
            time_elapsed = restore.write_srvideo(self.uscnn,lr_path,sr_path,self.upscaling_factor,print_frequency=print_frequency,crf=qp,fps=fps,gpu=gpu)
        elif(media_type == 'i'):
            time_elapsed = restore.write_sr_images(self.uscnn, lr_imagepath=lr_path, sr_imagepath=sr_path,scale=self.upscaling_factor)
        else:
            print(">> Media type not defined or not suported!")
            return 0
        return time_elapsed
    
    


def main():
    logging.basicConfig(filename='../logs/cisrdcnn.log', level=logging.INFO)
    logging.info('Started')

    #------------------------------------------------------

    # Instantiate the TSRGAN object
    logging.info(">> Creating the CISRDCNN network")
    cisrdcnn = CISRDCNN(height_lr=16, width_lr=16,lr=1e-3,upscaling_factor=2,channels=3,colorspace = 'RGB')
    #cisrdcnn.load_weights(weights='../model/CISRDCNN_v1_2X.h5')


    """ datapath = '../../data/videoset/540p/' 
    outpath = '../out/540p_2X/'
    for dirpath, _, filenames in os.walk(datapath):
        for filename in [f for f in sorted(filenames) if any(filetype in f.lower() for filetype in ['jpeg', 'png', 'jpg','mp4','264','webm','wma'])]:
            print(os.path.join(dirpath, filename),outpath+filename.split('.')[0]+'.mp4')
            t = cisrdcnn.predict(
                    lr_path=os.path.join(dirpath, filename), 
                    sr_path=outpath+filename.split('.')[0]+'.mp4',
                    qp=0,
                    media_type='v',
                    gpu=False
                ) """

    """ datapath = '../../data/videoset/360p/' 
    outpath = '../out/360p_2X/'
    i=1
    for dirpath, _, filenames in os.walk(datapath):
        for filename in [f for f in sorted(filenames) if any(filetype in f.lower() for filetype in ['jpeg', 'png', 'jpg','mp4','264','webm','wma'])]:
            if(i==17):
                print(os.path.join(dirpath, filename),outpath+filename.split('.')[0]+'.mp4')
                t = cisrdcnn.predict(
                        lr_path=os.path.join(dirpath, filename), 
                        sr_path=outpath+filename.split('.')[0]+'.mp4',
                        qp=0,
                        media_type='v',
                        gpu=False
                    )
            i+=1 """
                

    cisrdcnn.train_uscnn(
            epochs=10000,
            batch_size=64,
            steps_per_epoch=10,
            steps_per_validation=5,
            crops_per_image=4,
            print_frequency=5,
            log_tensorboard_update_freq=10,
            workers=1,
            max_queue_size=10,
            model_name='CISRDCNN',
            datapath_train='../../../Documents/data/train_large/data_large/', 
            datapath_validation='../../data/val_large/', 
            datapath_test='../../data/benchmarks/Set5/',
            log_weight_path='../model/', 
            log_tensorboard_path='../logs/',
            log_test_path='../test/'
    )

    #------------------------------------------------------


    logging.info('Finished')




# Run the CISRDCNN network
if __name__ == "__main__":
    main()

    