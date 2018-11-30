from chunkflow.chunk_operations.chunk_operation import OffProcessChunkOperation
from chunkflow.chunk_operations.inference.pytorch_inference import PyTorchInference
from chunkflow.chunk_operations.inference_operation import IdentityInferenceOperation, InferenceFactory


class TestInferenceOperation:
    def test_weight_mapping_2d(self, chunk_datasource_manager, block_datasource_manager):
        patch_shape = (5, 10, 10)

        inference_factory = InferenceFactory(tuple(patch_shape), channel_dimensions=(1,))

        model_path = None
        checkpoint_path = None
        identity_operation = inference_factory.get_operation('identity', model_path, checkpoint_path)
        assert isinstance(identity_operation, IdentityInferenceOperation)

        identity_operation = inference_factory.get_operation('identity', model_path, checkpoint_path,
                                                             off_main_process=True)
        assert isinstance(identity_operation, OffProcessChunkOperation)

        identity_operation = inference_factory.get_operation('pytorch', model_path, checkpoint_path)
        assert isinstance(identity_operation, PyTorchInference)

        identity_operation = inference_factory.get_operation('pytorch', model_path, checkpoint_path,
                                                             off_main_process=True)
        assert isinstance(identity_operation, OffProcessChunkOperation)
