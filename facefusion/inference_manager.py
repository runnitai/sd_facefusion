from functools import lru_cache
from time import sleep
from typing import List

import onnx
from onnxruntime import InferenceSession

from facefusion import process_manager, state_manager
from facefusion.app_context import detect_app_context
from facefusion.execution import create_execution_providers, has_execution_provider
from facefusion.thread_helper import thread_lock
from facefusion.typing import DownloadSet, ExecutionProviderKey, InferencePool, InferencePoolSet, ModelInitializer

INFERENCE_POOLS: InferencePoolSet = \
    {
        'cli': {},  # type:ignore[typeddict-item]
        'ui': {}  # type:ignore[typeddict-item]
    }


def get_inference_pool(model_context: str, model_sources: DownloadSet, preferred_provider: str = "default") -> InferencePool:
    global INFERENCE_POOLS

    with thread_lock():
        while process_manager.is_checking():
            sleep(0.5)
        app_context = detect_app_context()
        inference_context = get_inference_context(model_context, preferred_provider)
        requested_context = INFERENCE_POOLS.get(app_context).get(inference_context)
        if app_context == 'cli' and INFERENCE_POOLS.get('ui').get(inference_context) and not requested_context:
            INFERENCE_POOLS['cli'][inference_context] = INFERENCE_POOLS.get('ui').get(inference_context)
        if app_context == 'ui' and INFERENCE_POOLS.get('cli').get(inference_context) and not requested_context:
            INFERENCE_POOLS['ui'][inference_context] = INFERENCE_POOLS.get('cli').get(inference_context)
        if not requested_context:
            execution_provider_keys = resolve_execution_provider_keys(model_context, preferred_provider)
            print(f"Creating inference pool for {model_context} with {execution_provider_keys}.")
            INFERENCE_POOLS[app_context][inference_context] = create_inference_pool(model_sources,
                                                                                    state_manager.get_item(
                                                                                        'execution_device_id'),
                                                                                    execution_provider_keys)

        return INFERENCE_POOLS.get(app_context).get(inference_context)


def create_inference_pool(model_sources: DownloadSet, execution_device_id: str,
                          execution_provider_keys: List[ExecutionProviderKey]) -> InferencePool:
    inference_pool: InferencePool = {}

    for model_name in model_sources.keys():
        inference_pool[model_name] = create_inference_session(model_sources.get(model_name).get('path'),
                                                              execution_device_id, execution_provider_keys)
    return inference_pool


def clear_inference_pool(model_context: str, preferred_provider: str = "default") -> None:
    global INFERENCE_POOLS

    app_context = detect_app_context()
    inference_context = get_inference_context(model_context, preferred_provider)

    if INFERENCE_POOLS.get(app_context).get(inference_context):
        del INFERENCE_POOLS[app_context][inference_context]


def create_inference_session(model_path: str, execution_device_id: str,
                             execution_provider_keys: List[ExecutionProviderKey]) -> InferenceSession:
    execution_providers = create_execution_providers(execution_device_id, execution_provider_keys)
    return InferenceSession(model_path, providers=execution_providers)


@lru_cache(maxsize=None)
def get_static_model_initializer(model_path: str) -> ModelInitializer:
    model = onnx.load(model_path)
    return onnx.numpy_helper.to_array(model.graph.initializer[-1])


def resolve_execution_provider_keys(model_context: str, preferred_provider: str = "default") -> List[ExecutionProviderKey]:
    if has_execution_provider('coreml') and ('age_modifier' in model_context or 'frame_colorizer' in model_context):
        return ['cpu']
    if preferred_provider != "default" and has_execution_provider(preferred_provider):
        print(f"Using preferred provider {preferred_provider} for {model_context}.")
        return [preferred_provider]

    return state_manager.get_item('execution_providers')


def get_inference_context(model_context: str, preferred_provider: str = "default") -> str:
    execution_provider_keys = resolve_execution_provider_keys(model_context, preferred_provider)
    inference_context = model_context + '.' + '_'.join(execution_provider_keys)
    return inference_context
