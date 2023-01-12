# TFRecordHandler
A package including functions to write and read TFRecord files for Tensorflow

## Example
### Write TFRecord file
```Python
data_fields = {
    'features':features,
    'label':label,
}
write_to_tfrecord(data_fields, './training.tfrecord')
```

### Read TFRecord file
```Python
feature_info = {
    'features': (tf.float32, [25,]),
    'label': (tf.int32, [2,])
    }

dataset = read_tfrecord('./training.tfrecord', feature_info)
```
