import gradio
from typing import Optional

from facefusion import state_manager, wording
from facefusion.processors.classes.lip_syncer import LipSyncer
from facefusion.processors.typing import LipSyncerModel
from facefusion.uis.core import register_ui_component, get_ui_component

LIP_SYNCER_MODEL_DROPDOWN: Optional[gradio.Dropdown] = None
LIP_SYNC_EMPTY_AUDIO_CHECKBOX: Optional[gradio.Checkbox] = None
PROCESSOR_KEY = 'Lip Syncer'


def render() -> None:
    global LIP_SYNCER_MODEL_DROPDOWN, LIP_SYNC_EMPTY_AUDIO_CHECKBOX

    LIP_SYNCER_MODEL_DROPDOWN = gradio.Dropdown(
        label=wording.get('uis.lip_syncer_model_dropdown'),
        choices=['musetalk_v15'],  # Only MuseTalk now
        value='musetalk_v15',  # Fixed default value
        visible=PROCESSOR_KEY in state_manager.get_item('processors')
    )
    
    LIP_SYNC_EMPTY_AUDIO_CHECKBOX = gradio.Checkbox(
        label='Sync Empty Audio',
        info='Generate lip movements even when no audio is detected',
        value=False,  # Default to False
        visible=PROCESSOR_KEY in state_manager.get_item('processors')
    )
    
    register_ui_component('lip_syncer_model_dropdown', LIP_SYNCER_MODEL_DROPDOWN)
    register_ui_component('lip_sync_empty_audio_checkbox', LIP_SYNC_EMPTY_AUDIO_CHECKBOX)


def listen() -> None:
    if LIP_SYNCER_MODEL_DROPDOWN:
        LIP_SYNCER_MODEL_DROPDOWN.change(update_lip_syncer_model, inputs=LIP_SYNCER_MODEL_DROPDOWN,
                                         outputs=LIP_SYNCER_MODEL_DROPDOWN)
    
    if LIP_SYNC_EMPTY_AUDIO_CHECKBOX:
        LIP_SYNC_EMPTY_AUDIO_CHECKBOX.change(update_lip_sync_empty_audio, inputs=LIP_SYNC_EMPTY_AUDIO_CHECKBOX)

    # Safely handle processor checkbox group
    try:
        processors_checkbox_group = get_ui_component('processors_checkbox_group')
        if processors_checkbox_group:
            processors_checkbox_group.change(remote_update, inputs=processors_checkbox_group,
                                             outputs=[LIP_SYNCER_MODEL_DROPDOWN, LIP_SYNC_EMPTY_AUDIO_CHECKBOX])
    except:
        # Fallback if processors checkbox group isn't available yet
        pass


def remote_update(processors) -> tuple:
    has_lip_syncer = 'Lip Syncer' in processors
    return gradio.update(visible=has_lip_syncer), gradio.update(visible=has_lip_syncer)


def update_lip_syncer_model(lip_syncer_model: LipSyncerModel) -> gradio.update:
    try:
        state_manager.set_item('lip_syncer_model', lip_syncer_model)
        return gradio.update(value=lip_syncer_model)
    except:
        return gradio.update()


def update_lip_sync_empty_audio(lip_sync_empty_audio: bool) -> None:
    state_manager.set_item('lip_sync_empty_audio', lip_sync_empty_audio)
