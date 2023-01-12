import tensorflow as tf
import numpy as np

def write_to_tfrecord(data_fields, out_file):
    """
    Writes data and labels to a TFRecord file
    Args:
        data_fields: a dictionary of data fields to be written, key is name of the field and value is the data
        labels: list of labels corresponding to the data
        out_file: name of the output TFRecord file
    """
    n = len(data_fields[list(data_fields.keys())[0]])
    with tf.io.TFRecordWriter(out_file) as writer:
        for i in range(n):
            features = {}
            for field_name, field_data in data_fields.items():
                if type(field_data) == np.ndarray:
                    data_type = str(field_data.dtype)
                else:
                    data_type = str(type(field_data[0]))
                
                if "float" in data_type:
                    features[field_name] = tf.train.Feature(float_list=tf.train.FloatList(value=field_data[i]))
                elif "int" in data_type:
                    features[field_name] = tf.train.Feature(int64_list=tf.train.Int64List(value=field_data[i]))
                elif "str" in data_type:
                    features[field_name] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[d.encode() for d in field_data[i]]))
                else:
                    raise ValueError(f"Unsupported data type: {data_type}")
            
            example = tf.train.Example(
                features=tf.train.Features(
                    feature=features
                )
            )
            writer.write(example.SerializeToString())
            
def read_tfrecord(file_path, fields_info):
    """
    Reads data and labels from a TFRecord file
    Args:
        file_path: path to the TFRecord file
        fields_info: a dictionary of fields, key is name of the field and value is a tuple of (data_type, shape)
    Returns:
        dataset: a Tensorflow Dataset
    """
    dataset = tf.data.TFRecordDataset(file_path)
    feature_description = {}

    for name, tup in fields_info.items():
      dtype = tup[0]
      shape = tup[1]
      feature_description[name] = tf.io.FixedLenFeature(shape, dtype)

    def _parse_tfrecord_function(example_proto):
      # Parse the input tf.train.Example proto using the dictionary above.
      return tf.io.parse_single_example(example_proto, feature_description)

    dataset = dataset.map(_parse_tfrecord_function)
    return dataset
