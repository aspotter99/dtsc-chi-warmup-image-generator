# Importing Images with Keras Image Generator

Image data is a somewhat new concept, and is  certainly formatted differently than most data we have worked with so far. 

Luckily keras provides us with some useful tools for easily importing image datasets.


```python
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Turn of TensorFlow deprecation warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Modeling
from keras.models import Sequential 
from keras.layers import Dense 
```

    Using TensorFlow backend.
    /Users/joel/opt/anaconda3/envs/mlearn/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:516: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint8 = np.dtype([("qint8", np.int8, 1)])
    /Users/joel/opt/anaconda3/envs/mlearn/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:517: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
    /Users/joel/opt/anaconda3/envs/mlearn/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:518: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint16 = np.dtype([("qint16", np.int16, 1)])
    /Users/joel/opt/anaconda3/envs/mlearn/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:519: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
    /Users/joel/opt/anaconda3/envs/mlearn/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:520: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint32 = np.dtype([("qint32", np.int32, 1)])
    /Users/joel/opt/anaconda3/envs/mlearn/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:525: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      np_resource = np.dtype([("resource", np.ubyte, 1)])
    /Users/joel/opt/anaconda3/envs/mlearn/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:541: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint8 = np.dtype([("qint8", np.int8, 1)])
    /Users/joel/opt/anaconda3/envs/mlearn/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:542: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
    /Users/joel/opt/anaconda3/envs/mlearn/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:543: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint16 = np.dtype([("qint16", np.int16, 1)])
    /Users/joel/opt/anaconda3/envs/mlearn/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:544: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
    /Users/joel/opt/anaconda3/envs/mlearn/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:545: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint32 = np.dtype([("qint32", np.int32, 1)])
    /Users/joel/opt/anaconda3/envs/mlearn/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:550: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      np_resource = np.dtype([("resource", np.ubyte, 1)])


In this repo, our image data is stored a like so:

```
data/training_set/cat
                    |__cat.1.jpg
                    |__...
                  dog
                    |__dog.1.jpg
                    |__...
data/test_set/cat
                |__cat.4001.jpg
                |__...
              dog
                |__dog.4001.jpg
                |__...
```
The organization is such that our training and test set folders both contain two folders. 1 for each class.  
We can easily import these images, by using a Keras ImageDataGenerator. 

ImageDataGenerator objects have several parameters that allow us to rotate, flip, or shift images which can improve the likelihood of our model generalizing to new data. We will not these features this morning, but we  will use the rescale parameter to normalize the image's pixel values.


```python
data_generator = ImageDataGenerator(rescale=1.0/256.0)
```

Now that we've instantiated an `ImageDataGenerator` object, we can create generators for the training and testing data.

Now we can create a basic CNN model.

And then instead of using `model.fit`, we use `model.fit_generator` and instead of a train split, we use the generator for our training data.

>This single epoch took about 15 minutes to run.
