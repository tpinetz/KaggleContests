import numpy as np


class SampleTransformation:
    # A sample transformation.

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        pass


class IdentityTransformation(SampleTransformation):
    # A transformation that does not do anything.

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        return sample


class FloatCastTransformation(SampleTransformation):
    # Casts the sample datatype to single-precision float (e.g. numpy.float32).

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        return sample.astype(np.float32)


class SubtractionTransformation(SampleTransformation):
    # Subtract a scalar from all features.

    @staticmethod
    def from_dataset_mean(dataset, tform=None):
        # Return a transformation that will subtract by the global mean
        # over all samples and features in a dataset.
        # tform is an optional SampleTransformation to apply before computation.
        return SubtractionTransformation(
                np.mean([sample if tform is None else tform.apply(sample) for sample in dataset.data]))

    def __init__(self, value):
        # Constructor.
        # value is a scalar to subtract.
        self.min_value = value

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        # The sample datatype must be single-precision float.
        return (sample - self.min_value).astype(sample.dtype)

    def value(self):
        # Return the subtracted value.
        return self.min_value


class DivisionTransformation(SampleTransformation):
    # Divide all features by a scalar.

    @staticmethod
    def from_dataset_stddev(dataset, tform=None):
        # Return a transformation that will divide by the global standard deviation
        # over all samples and features in a dataset.
        # tform is an optional SampleTransformation to apply before computation.
        return DivisionTransformation(
                np.std([sample if tform is None else tform.apply(sample) for sample in dataset.data]))

    def __init__(self, value):
        # Constructor.
        # value is a scalar divisor != 0.
        if value != 0:
            self.div_value = value
        else:
            raise Exception()

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        # The sample datatype must be single-precision float.
        return (sample / self.div_value).astype(sample.dtype)

    def value(self):
        # Return the divisor.
        return self.div_value


class PerChannelSubtractionImageTransformation(SampleTransformation):
    # Perform per-channel subtraction of of image samples with a scalar.

    @staticmethod
    def from_dataset_mean(dataset, tform=None):
        # Return a transformation that will subtract by the global mean
        # over all samples and features in a dataset, independently for every color channel.
        # tform is an optional SampleTransformation to apply before computation.
        # samples must be 3D tensors with shape [rows,cols,channels].
        # rows, cols, channels can be arbitrary values > 0.
        PerChannelSubtractionImageTransformation(
           [np.mean(dataset.data[:, :, :, i]) for i in range(0, dataset.data.shape[3])]
        )

    def __init__(self, values):
        # Constructor.
        # values is a vector of c values to subtract, one per channel.
        # c can be any value > 0.
        if np.any(self.values <= 0):
            raise Exception()
        self.values = values

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        # sample must be a 3D tensor with shape [rows,cols,c].
        # The sample datatype must be single-precision float.
        return sample - self.values

    def values(self):
        # Return the subtracted values.
        return self.values


class PerChannelDivisionImageTransformation(SampleTransformation):
    # Perform per-channel division of of image samples with a scalar.

    @staticmethod
    def from_dataset_stddev(dataset, tform=None):
        # Return a transformation that will divide by the global standard deviation
        # over all samples and features in a dataset, independently for every color channel.
        # tform is an optional SampleTransformation to apply before computation.
        # samples must be 3D tensors with shape [rows,cols,channels].
        # rows, cols, channels can be arbitrary values > 0.
        PerChannelDivisionImageTransformation(
           [np.std(dataset.data[:, :, :, i]) for i in range(0, dataset.data.shape[3])]
        )

    def __init__(self, values):
        # Constructor.
        # values is a vector of c divisors, one per channel.
        # c can be any value > 0.
        if np.any(self.values <= 0):
            raise Exception()
        self.values = values

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        # sample must be a 3D tensor with shape [rows,cols,c].
        # The sample datatype must be single-precision float.
        return sample / self.values

    def values(self):
        # Return the divisors.
        return self.values


class TransformationSequence(SampleTransformation):
    # Applies a sequence of transformations
    # in the order they were added via add_transformation().
    transformations = []

    def add_transformation(self, transformation):
        # Add a transformation (type SampleTransformation) to the sequence.
        self.transformations.append(transformation)

    def get_transformation(self, tid):
        # Return the id-th transformation added via add_transformation.
        # The first transformation added has ID 0.
        self.transformations[tid]

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        result = sample.copy()
        for transformation in self.transformations:
            result = transformation.apply(result)
        return result
