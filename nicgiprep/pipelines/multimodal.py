from nicgiprep.pipelines.base import Processor


class MMProcessor(Processor):
    """Base class for multi-modal processing pipelines.

    Stub — extend to add modality-specific logic.
    """
    pass


class MultiMRIProcessor(MMProcessor):
    """Multi-MRI processing pipeline combining several MRI contrasts.

    Stub — extend to implement cross-modal registration and fusion.
    """
    pass