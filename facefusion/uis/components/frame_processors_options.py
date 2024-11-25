import gradio
from typing import List, Optional, Tuple

import facefusion.globals
from facefusion import wording
from facefusion.processors.frame import globals as frame_processors_globals, choices as frame_processors_choices
from facefusion.processors.frame.core import load_frame_processor_module
from facefusion.processors.frame.typings import FaceDebuggerItem, FaceEnhancerModel, FaceSwapperModel, \
    FrameEnhancerModel, LipSyncerModel
from facefusion.uis.core import get_ui_component, register_ui_component
from facefusion.uis.components.source import update as update_source, update_2 as update_source_2, \
    check_swap_source_style
from facefusion.uis.typing import File

FACE_DEBUGGER_ITEMS_CHECKBOX_GROUP: Optional[gradio.CheckboxGroup] = None
FACE_ENHANCER_MODEL_DROPDOWN: Optional[gradio.Dropdown] = None
FACE_ENHANCER_BLEND_SLIDER: Optional[gradio.Slider] = None
FACE_SWAPPER_MODEL_DROPDOWN: Optional[gradio.Dropdown] = None
FACE_SWAPPER_WEIGHT_SLIDER: Optional[gradio.Slider] = None
FRAME_ENHANCER_MODEL_DROPDOWN: Optional[gradio.Dropdown] = None
FRAME_ENHANCER_BLEND_SLIDER: Optional[gradio.Slider] = None
LIP_SYNCER_MODEL_DROPDOWN: Optional[gradio.Dropdown] = None
STYLE_CHANGER_MODEL_DROPDOWN: Optional[gradio.Dropdown] = None
STYLE_TARGET_RADIO: Optional[gradio.Radio] = None


def render() -> None:
    global FACE_DEBUGGER_ITEMS_CHECKBOX_GROUP
    global FACE_ENHANCER_MODEL_DROPDOWN
    global FACE_ENHANCER_BLEND_SLIDER
    global FACE_SWAPPER_MODEL_DROPDOWN
    global FACE_SWAPPER_WEIGHT_SLIDER
    global FRAME_ENHANCER_MODEL_DROPDOWN
    global FRAME_ENHANCER_BLEND_SLIDER
    global LIP_SYNCER_MODEL_DROPDOWN
    FACE_DEBUGGER_ITEMS_CHECKBOX_GROUP = gradio.CheckboxGroup(
        label=wording.get('uis.face_debugger_items_checkbox_group'),
        choices=frame_processors_choices.face_debugger_items,
        value=frame_processors_globals.face_debugger_items,
        visible='face_debugger' in facefusion.globals.frame_processors,
        elem_id='face_debugger_items_checkbox'
    )
    FACE_SWAPPER_WEIGHT_SLIDER = gradio.Slider(
        label=wording.get('uis.face_swapper_weight_slider'),
        value=frame_processors_globals.face_swapper_weight,
        step=frame_processors_choices.face_swapper_weight_range[1] - frame_processors_choices.face_swapper_weight_range[
            0],
        minimum=frame_processors_choices.face_swapper_weight_range[0],
        maximum=frame_processors_choices.face_swapper_weight_range[-1],
        visible='face_swapper' in facefusion.globals.frame_processors
    )
    FACE_ENHANCER_MODEL_DROPDOWN = gradio.Dropdown(
        label=wording.get('uis.face_enhancer_model_dropdown'),
        choices=frame_processors_choices.face_enhancer_models,
        value=frame_processors_globals.face_enhancer_model,
        visible='face_enhancer' in facefusion.globals.frame_processors,
        elem_id='face_enhancer_model_dropdown'
    )
    FACE_ENHANCER_BLEND_SLIDER = gradio.Slider(
        label=wording.get('uis.face_enhancer_blend_slider'),
        value=frame_processors_globals.face_enhancer_blend,
        step=frame_processors_choices.face_enhancer_blend_range[1] - frame_processors_choices.face_enhancer_blend_range[
            0],
        minimum=frame_processors_choices.face_enhancer_blend_range[0],
        maximum=frame_processors_choices.face_enhancer_blend_range[-1],
        visible='face_enhancer' in facefusion.globals.frame_processors,
        elem_id='face_enhancer_blend_slider'
    )
    FACE_SWAPPER_MODEL_DROPDOWN = gradio.Dropdown(
        label=wording.get('uis.face_swapper_model_dropdown'),
        choices=frame_processors_choices.face_swapper_models,
        value=frame_processors_globals.face_swapper_model,
        visible='face_swapper' in facefusion.globals.frame_processors
    )
    FRAME_ENHANCER_MODEL_DROPDOWN = gradio.Dropdown(
        label=wording.get('uis.frame_enhancer_model_dropdown'),
        choices=frame_processors_choices.frame_enhancer_models,
        value=frame_processors_globals.frame_enhancer_model,
        visible='frame_enhancer' in facefusion.globals.frame_processors,
        elem_id='frame_enhancer_model_dropdown'
    )
    FRAME_ENHANCER_BLEND_SLIDER = gradio.Slider(
        label=wording.get('uis.frame_enhancer_blend_slider'),
        value=frame_processors_globals.frame_enhancer_blend,
        step=frame_processors_choices.frame_enhancer_blend_range[1] -
             frame_processors_choices.frame_enhancer_blend_range[0],
        minimum=frame_processors_choices.frame_enhancer_blend_range[0],
        maximum=frame_processors_choices.frame_enhancer_blend_range[-1],
        visible='face_enhancer' in facefusion.globals.frame_processors,
        elem_id='frame_enhancer_blend_slider'
    )
    LIP_SYNCER_MODEL_DROPDOWN = gradio.Dropdown(
        label=wording.get('uis.lip_syncer_model_dropdown'),
        choices=frame_processors_choices.lip_syncer_models,
        value=frame_processors_globals.lip_syncer_model,
        visible='lip_syncer' in facefusion.globals.frame_processors,
        elem_id='lip_syncer_model'
    )
    STYLE_CHANGER_MODEL_DROPDOWN = gradio.Dropdown(
        label=wording.get('uis.style_changer_model_dropdown'),
        choices=frame_processors_choices.style_changer_models,
        value=frame_processors_globals.style_changer_model,
        visible='style_changer' in facefusion.globals.frame_processors,
        elem_id='style_changer_model_dropdown'
    )
    STYLE_TARGET_RADIO = gradio.Radio(
        label=wording.get('uis.style_target_radio'),
        choices=["source", "target"],
        value=frame_processors_globals.style_changer_target,
        visible='style_changer' in facefusion.globals.frame_processors,
        elem_id='style_target_radio'
    )
    register_ui_component('face_debugger_items_checkbox_group', FACE_DEBUGGER_ITEMS_CHECKBOX_GROUP)
    register_ui_component('face_enhancer_model_dropdown', FACE_ENHANCER_MODEL_DROPDOWN)
    register_ui_component('face_enhancer_blend_slider', FACE_ENHANCER_BLEND_SLIDER)
    register_ui_component('face_swapper_model_dropdown', FACE_SWAPPER_MODEL_DROPDOWN)
    register_ui_component('face_swapper_weight_slider', FACE_SWAPPER_WEIGHT_SLIDER)
    register_ui_component('frame_enhancer_model_dropdown', FRAME_ENHANCER_MODEL_DROPDOWN)
    register_ui_component('frame_enhancer_blend_slider', FRAME_ENHANCER_BLEND_SLIDER)
    register_ui_component('lip_syncer_model_dropdown', LIP_SYNCER_MODEL_DROPDOWN)
    register_ui_component('style_changer_model_dropdown', STYLE_CHANGER_MODEL_DROPDOWN)
    register_ui_component('style_target_radio', STYLE_TARGET_RADIO)


def listen() -> None:
    FACE_DEBUGGER_ITEMS_CHECKBOX_GROUP.change(update_face_debugger_items, inputs=FACE_DEBUGGER_ITEMS_CHECKBOX_GROUP)
    FACE_ENHANCER_MODEL_DROPDOWN.change(update_face_enhancer_model, inputs=FACE_ENHANCER_MODEL_DROPDOWN,
                                        outputs=FACE_ENHANCER_MODEL_DROPDOWN)
    FACE_ENHANCER_BLEND_SLIDER.change(update_face_enhancer_blend, inputs=FACE_ENHANCER_BLEND_SLIDER)
    FACE_SWAPPER_MODEL_DROPDOWN.change(update_face_swapper_model, inputs=FACE_SWAPPER_MODEL_DROPDOWN,
                                       outputs=FACE_SWAPPER_MODEL_DROPDOWN)
    FACE_SWAPPER_WEIGHT_SLIDER.change(update_face_swapper_weight, inputs=FACE_SWAPPER_WEIGHT_SLIDER)
    FRAME_ENHANCER_MODEL_DROPDOWN.change(update_frame_enhancer_model, inputs=FRAME_ENHANCER_MODEL_DROPDOWN,
                                         outputs=FRAME_ENHANCER_MODEL_DROPDOWN)
    FRAME_ENHANCER_BLEND_SLIDER.change(update_frame_enhancer_blend, inputs=FRAME_ENHANCER_BLEND_SLIDER)
    LIP_SYNCER_MODEL_DROPDOWN.change(update_lip_syncer_model, inputs=LIP_SYNCER_MODEL_DROPDOWN,
                                     outputs=LIP_SYNCER_MODEL_DROPDOWN)
    STYLE_CHANGER_MODEL_DROPDOWN.change(update_style_changer_model, inputs=STYLE_CHANGER_MODEL_DROPDOWN,
                                        outputs=STYLE_CHANGER_MODEL_DROPDOWN)
    target_file = get_ui_component('target_file')
    target_file_2 = get_ui_component('target_file_2')
    STYLE_TARGET_RADIO.change(update_style_target, inputs=[STYLE_TARGET_RADIO, target_file, target_file_2], outputs=[target_file, target_file_2])
    frame_processors_checkbox_group = get_ui_component('frame_processors_checkbox_group')
    if frame_processors_checkbox_group:
        frame_processors_checkbox_group.change(update_frame_processors, inputs=frame_processors_checkbox_group,
                                               outputs=[FACE_DEBUGGER_ITEMS_CHECKBOX_GROUP,
                                                        FACE_ENHANCER_MODEL_DROPDOWN, FACE_ENHANCER_BLEND_SLIDER,
                                                        FACE_SWAPPER_MODEL_DROPDOWN, FACE_SWAPPER_WEIGHT_SLIDER,
                                                        FRAME_ENHANCER_MODEL_DROPDOWN, FRAME_ENHANCER_BLEND_SLIDER,
                                                        LIP_SYNCER_MODEL_DROPDOWN])


def update_frame_processors(frame_processors: List[str]) -> Tuple[
    gradio.CheckboxGroup, gradio.Dropdown, gradio.Slider, gradio.Dropdown, gradio.Slider, gradio.Dropdown, gradio.Slider, gradio.Dropdown]:
    has_face_debugger = 'face_debugger' in frame_processors
    has_face_enhancer = 'face_enhancer' in frame_processors
    has_face_swapper = 'face_swapper' in frame_processors
    has_frame_enhancer = 'frame_enhancer' in frame_processors
    has_lip_syncer = 'lip_syncer' in frame_processors
    return gradio.update(visible=has_face_debugger), gradio.update(visible=has_face_enhancer), gradio.update(
        visible=has_face_enhancer), gradio.update(visible=has_face_swapper), gradio.update(
        visible=has_face_swapper), gradio.update(visible=has_frame_enhancer), gradio.update(
        visible=has_frame_enhancer), gradio.update(visible=has_lip_syncer)


def update_face_debugger_items(face_debugger_items: List[FaceDebuggerItem]) -> None:
    frame_processors_globals.face_debugger_items = face_debugger_items


def update_face_swapper_weight(face_swapper_weight: float) -> None:
    frame_processors_globals.face_swapper_weight = face_swapper_weight


def update_face_enhancer_model(face_enhancer_model: FaceEnhancerModel) -> gradio.Dropdown:
    frame_processors_globals.face_enhancer_model = face_enhancer_model
    face_enhancer_module = load_frame_processor_module('face_enhancer')
    face_enhancer_module.clear_frame_processor()
    face_enhancer_module.set_options('model', face_enhancer_module.MODELS[face_enhancer_model])
    if not face_enhancer_module.pre_check():
        return gradio.update(visible=True)
    return gradio.update(value=face_enhancer_model)


def update_face_enhancer_blend(face_enhancer_blend: int) -> None:
    frame_processors_globals.face_enhancer_blend = face_enhancer_blend


def update_face_swapper_model(face_swapper_model: FaceSwapperModel) -> gradio.update:
    frame_processors_globals.face_swapper_model = face_swapper_model
    if face_swapper_model == 'blendswap_256':
        facefusion.globals.face_recognizer_model = 'arcface_blendswap'
    if face_swapper_model == 'inswapper_128' or face_swapper_model == 'inswapper_128_fp16':
        facefusion.globals.face_recognizer_model = 'arcface_inswapper'
    if face_swapper_model == 'simswap_256' or face_swapper_model == 'simswap_512_unofficial':
        facefusion.globals.face_recognizer_model = 'arcface_simswap'
    if face_swapper_model == 'uniface_256':
        facefusion.globals.face_recognizer_model = 'arcface_uniface'
    face_swapper_module = load_frame_processor_module('face_swapper')
    face_swapper_module.clear_frame_processor()
    face_swapper_module.set_options('model', face_swapper_module.MODELS[face_swapper_model])
    if not face_swapper_module.pre_check():
        return gradio.update()
    return gradio.update(value=face_swapper_model)


def update_frame_enhancer_model(frame_enhancer_model: FrameEnhancerModel) -> gradio.update:
    frame_processors_globals.frame_enhancer_model = frame_enhancer_model
    frame_enhancer_module = load_frame_processor_module('frame_enhancer')
    frame_enhancer_module.clear_frame_processor()
    frame_enhancer_module.set_options('model', frame_enhancer_module.MODELS[frame_enhancer_model])
    if not frame_enhancer_module.pre_check():
        return gradio.update()
    return gradio.update(value=frame_enhancer_model)


def update_frame_enhancer_blend(frame_enhancer_blend: int) -> None:
    frame_processors_globals.frame_enhancer_blend = frame_enhancer_blend


def update_lip_syncer_model(lip_syncer_model: LipSyncerModel) -> gradio.Dropdown:
    frame_processors_globals.lip_syncer_model = lip_syncer_model
    lip_syncer_module = load_frame_processor_module('lip_syncer')
    lip_syncer_module.clear_frame_processor()
    lip_syncer_module.set_options('model', lip_syncer_module.MODELS[lip_syncer_model])
    if lip_syncer_module.pre_check():
        return gradio.Dropdown(value=lip_syncer_model)
    return gradio.Dropdown()


def update_style_changer_model(style_changer_model: str) -> gradio.Dropdown:
    frame_processors_globals.style_changer_model = style_changer_model
    facefusion.globals.style_changer_model = style_changer_model
    style_changer_module = load_frame_processor_module('style_changer')
    style_changer_module.clear_frame_processor()
    style_changer_module.set_options('model', style_changer_model)
    if style_changer_module.pre_check():
        return gradio.Dropdown(value=style_changer_model)
    return gradio.Dropdown()


def update_style_target(style_target: str, source_file: List[File], source_file_2: List[File]) -> Tuple[gradio.update, gradio.update]:
    frame_processors_globals.style_changer_target = style_target
    facefusion.globals.style_changer_target = style_target
    style_changer_module = load_frame_processor_module('style_changer')
    style_changer_module.set_options('target', style_target)
    return gradio.update(check_swap_source_style(source_file)), gradio.update(check_swap_source_style(source_file_2))


