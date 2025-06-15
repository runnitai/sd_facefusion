import gradio

from facefusion import state_manager
from facefusion.uis.components import (age_modifier_options, common_options, expression_restorer_options, \
                                       face_debugger_options, face_detector, face_editor_options, \
                                       face_enhancer_options, face_landmarker, face_masker, face_selector,
                                       face_swapper_options, frame_colorizer_options, \
                                       frame_enhancer_options, instant_runner, job_manager, job_runner,
                                       lip_syncer_options, style_changer_options, \
                                       style_transfer_options, output, \
                                       output_options, \
                                       preview, processors, source, target, temp_frame, trim_frame, execution,
                                       execution_thread_count, execution_queue_count, ui_workflow)


def pre_check() -> bool:
    return True


def render() -> gradio.Blocks:
    with gradio.Blocks() as layout:
        # RunDiffusion Branding Header
        gradio.HTML('''
            <div style="text-align: center; margin-bottom: 2rem; padding: 1.5rem; background: linear-gradient(135deg, #ff6b35 0%, #004d7a 100%); border-radius: 16px; color: white; box-shadow: 0 4px 12px rgba(255, 107, 53, 0.2);">
                <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">
                    🎭 RD FaceFusion
                </h1>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
                    Professional Face Processing Suite - <strong>RunDiffusion Style Compliant</strong>
                </p>
                <div style="margin-top: 0.5rem; font-size: 0.9rem; opacity: 0.8;">
                    ✨ Workflow-Based Interface | 🎨 Modern Design | ⚡ Enhanced Performance
                </div>
            </div>
        ''', elem_classes=['rd-brand-header'])
        
        # Header with main workflow tabs
        with gradio.Tabs(elem_id="main_workflow_tabs") as main_tabs:
            
            # 1. SETUP TAB - Source and target selection
            with gradio.Tab("🎯 Setup", elem_id="setup_tab"):
                with gradio.Row():
                    with gradio.Column(scale=1):
                        with gradio.Group():
                            gradio.Markdown("### 📁 Source Content")
                            with gradio.Blocks():
                                source.render()
                    
                    with gradio.Column(scale=1):
                        with gradio.Group():
                            gradio.Markdown("### 🎬 Target Content")  
                            with gradio.Blocks():
                                target.render()
                                
                with gradio.Row():
                    with gradio.Column():
                        with gradio.Group():
                            gradio.Markdown("### ⚙️ Quick Start")
                            with gradio.Blocks():
                                processors.render()
                            with gradio.Blocks():
                                ui_workflow.render()
                                instant_runner.render()
            
            # 2. PREVIEW TAB - Enhanced timeline and preview
            with gradio.Tab("📺 Preview & Timeline", elem_id="preview_tab"):
                with gradio.Row():
                    with gradio.Column(scale=3):
                        with gradio.Group():
                            gradio.Markdown("### 🎥 Video Preview")
                            with gradio.Blocks():
                                preview.render()
                            
                            gradio.Markdown("### ⏱️ Timeline Controls")
                            with gradio.Blocks():
                                trim_frame.render()
                    
                    with gradio.Column(scale=1):
                        with gradio.Group():
                            gradio.Markdown("### 🎛️ Preview Options")
                            with gradio.Blocks():
                                face_debugger_options.render()
                            with gradio.Blocks():
                                temp_frame.render()
            
            # 3. FACE MANAGEMENT TAB - All face-related controls
            with gradio.Tab("👤 Face Management", elem_id="face_tab"):
                with gradio.Row():
                    with gradio.Column(scale=2):
                        with gradio.Group():
                            gradio.Markdown("### 🎯 Face Selection & Matching")
                            with gradio.Blocks():
                                face_selector.render()
                    
                    with gradio.Column(scale=1):
                        with gradio.Group():
                            gradio.Markdown("### 🎭 Face Detection")
                            with gradio.Blocks():
                                face_detector.render()
                            with gradio.Blocks():
                                face_landmarker.render()
                        
                        with gradio.Group():
                            gradio.Markdown("### 🎨 Face Masking")
                            with gradio.Blocks():
                                face_masker.render()
            
            # 4. PROCESSORS TAB - All processing options organized
            with gradio.Tab("🔧 Processors", elem_id="processors_tab"):
                with gradio.Tabs():
                    # Face Processing
                    with gradio.Tab("👤 Face Effects"):
                        with gradio.Row():
                            with gradio.Column():
                                with gradio.Group():
                                    gradio.Markdown("### 🔄 Face Swapping")
                                    with gradio.Blocks():
                                        face_swapper_options.render()
                                
                                with gradio.Group():
                                    gradio.Markdown("### 🎨 Style & Expression") 
                                    with gradio.Blocks():
                                        style_changer_options.render()
                                    with gradio.Blocks():
                                        expression_restorer_options.render()
                            
                            with gradio.Column():
                                with gradio.Group():
                                    gradio.Markdown("### ✨ Face Enhancement")
                                    with gradio.Blocks():
                                        face_enhancer_options.render()
                                    with gradio.Blocks():
                                        face_editor_options.render()
                                
                                with gradio.Group():
                                    gradio.Markdown("### 🔄 Age & Sync")
                                    with gradio.Blocks():
                                        age_modifier_options.render()
                                    with gradio.Blocks():
                                        lip_syncer_options.render()
                    
                    # Frame Processing  
                    with gradio.Tab("🎬 Frame Effects"):
                        with gradio.Row():
                            with gradio.Column():
                                with gradio.Group():
                                    gradio.Markdown("### 🌈 Frame Enhancement")
                                    with gradio.Blocks():
                                        frame_enhancer_options.render()
                                    with gradio.Blocks():
                                        frame_colorizer_options.render()
                            
                            with gradio.Column():
                                with gradio.Group():
                                    gradio.Markdown("### 🎨 Style Transfer")
                                    with gradio.Blocks():
                                        style_transfer_options.render()
            
            # 5. EXECUTION TAB - Processing and output
            with gradio.Tab("🚀 Execute", elem_id="execute_tab"):
                with gradio.Row():
                    with gradio.Column(scale=1):
                        with gradio.Group():
                            gradio.Markdown("### ⚡ Execution Settings")
                            with gradio.Blocks():
                                execution.render()
                                execution_thread_count.render()
                                execution_queue_count.render()
                        
                        with gradio.Group():
                            gradio.Markdown("### 📤 Output Settings")
                            with gradio.Blocks():
                                output_options.render()
                            with gradio.Blocks():
                                common_options.render()
                    
                    with gradio.Column(scale=1):
                        with gradio.Group():
                            gradio.Markdown("### 🎯 Job Management")
                            with gradio.Blocks():
                                job_runner.render()
                                job_manager.render()
                        
                        with gradio.Group():
                            gradio.Markdown("### 📁 Output")
                            with gradio.Blocks():
                                output.render()

    return layout


def listen() -> None:
    processors.listen()
    age_modifier_options.listen()
    expression_restorer_options.listen()
    face_debugger_options.listen()
    face_editor_options.listen()
    face_enhancer_options.listen()
    face_swapper_options.listen()
    frame_colorizer_options.listen()
    frame_enhancer_options.listen()
    lip_syncer_options.listen()
    style_changer_options.listen()
    style_transfer_options.listen()
    execution.listen()
    execution_thread_count.listen()
    execution_queue_count.listen()
    # memory.listen()
    temp_frame.listen()
    output_options.listen()
    source.listen()
    target.listen()
    output.listen()
    #ui_workflow.listen()
    instant_runner.listen()
    job_runner.listen()
    job_manager.listen()
    # terminal.listen()
    preview.listen()
    trim_frame.listen()
    face_selector.listen()
    face_masker.listen()
    face_detector.listen()
    face_landmarker.listen()
    common_options.listen()


def run(ui: gradio.Blocks) -> None:
    ui.launch(favicon_path='facefusion.ico', inbrowser=state_manager.get_item('open_browser'))
