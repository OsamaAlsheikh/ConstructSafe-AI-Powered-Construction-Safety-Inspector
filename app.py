# app.py  [ROOT FOLDER]
# The full Gradio UI — the only file the user sees.
# Three tabs: Run Inspection | Dashboard | About
# Called with: python app.py   -> http://localhost:7860

import gradio as gr
import plotly.graph_objects as go
from pipeline import run_inspection
from database import get_dashboard_stats

CUSTOM_CSS = '''
.header-container { background:linear-gradient(135deg,#1a1a2e,#0f3460);
                    padding:2rem; border-radius:12px; margin-bottom:1.5rem; }
.header-title    { font-size:2.2rem; font-weight:700; color:#f5a623; margin:0; }
.header-subtitle { font-size:1rem;   color:#a0aec0;   margin-top:0.5rem; }
#inspect-btn { background:#f5a623!important; color:#1a1a2e!important;
               font-weight:700!important; height:54px!important; }
'''

def _violations_html(violations, risk_level):
    """Render colour-coded violation cards as HTML string."""
    if not violations:
        return '<div style="background:#f0fff4;padding:1.5rem;border-radius:8px;text-align:center;">''<h3 style="color:#276749;">No Violations Detected</h3></div>'
    sev_colors = {'Willful':('#fff5f5','#e53e3e','red'),
                  'Serious':('#fffaf0','#dd6b20','orange')}
    risk_color = {'CRITICAL':'#e53e3e','HIGH':'#dd6b20',
                  'MEDIUM':'#3182ce','LOW':'#38a169'}.get(risk_level,'#718096')
    parts = [f'<div style="border:2px solid {risk_color};padding:1rem;border-radius:8px;text-align:center;">'
             f'<strong>OVERALL RISK: {risk_level} -- {len(violations)} violation(s)</strong></div>']
    for i,v in enumerate(violations,1):
        sev = v.get('severity','Other-Than-Serious')
        bg,bc,_ = sev_colors.get(sev,('#ebf8ff','#3182ce','blue'))
        parts.append(f'<div style="background:{bg};border:1px solid {bc};'
                     f'border-radius:8px;padding:1rem;margin-top:0.75rem;">'
                     f'<strong>#{i}: {v.get("violation_name",v.get("code",""))} [{sev}]</strong><br>'
                     f'Regulation: {v.get("regulation","N/A")}<br>'
                     f'Observation: {v.get("observation","N/A")}<br>'
                     f'Confidence: {v.get("confidence",0):.0%}</div>')
    return ''.join(parts)

def run_ui(media_file, site_name, supervisor_name, supervisor_email,
           progress=gr.Progress()):
    """Called when user clicks the inspection button."""
    if media_file is None or not site_name.strip():
        return '<p style="color:red;">Please upload a file and enter site name.</p>','','','',None,None
    progress(0.1, desc='Analysing video...')
    result = run_inspection(media_file, site_name.strip(),
                            supervisor_name.strip() or 'Site Supervisor',
                            supervisor_email.strip())
    progress(1.0, desc='Complete!')
    if not result['success']:
        return f"<p style='color:red;'>Error: {result['error']}</p>",'','','',None,None
    vr    = result['vision_results']
    notif = result['notification']
    t     = result['timing']
    timing_html = (f"<div style='background:#f7fafc;padding:0.75rem;border-radius:8px;font-size:0.85rem;'>"
                   f"Vision: {t.get('vision_seconds',0)}s | Report: {t.get('report_seconds',0)}s | "
                   f"Total: {t.get('total_seconds',0)}s | Inspection #{notif.get('inspection_id','?')}</div>")
    repeat_html = ''
    if notif.get('is_repeat_site'):
        repeat_html = (f"<div style='background:#fff5f5;border:2px solid #e53e3e;"
                       f"border-radius:8px;padding:1rem;margin-top:0.75rem;'>"
                       f"REPEAT OFFENDER: {notif['repeat_count']} violations in past 30 days.</div>")
    sample = result['sample_frames'][0] if result['sample_frames'] else None
    return (_violations_html(vr['violations'],vr['overall_risk_level'])+timing_html+repeat_html,
            result['report_text'],
            notif.get('email_subject',''), notif.get('email_body',''),
            sample)

def build_ui():
    with gr.Blocks(title='ConstructSafe',css=CUSTOM_CSS,
                   theme=gr.themes.Soft(primary_hue='amber')) as demo:
        gr.HTML('<div class="header-container">'
                '<h1 class="header-title">ConstructSafe</h1>'
                '<p class="header-subtitle">AI-Powered OSHA Safety Inspector | AMD MI300X</p>'
                '</div>')
        with gr.Tabs():
            with gr.TabItem('Run Inspection'):
                with gr.Row():
                    with gr.Column(scale=1):
                        media   = gr.File(label='Video or Image',
                                          file_types=['.mp4','.mov','.avi','.jpg','.png'])
                        site    = gr.Textbox(label='Site Name')
                        sup_n   = gr.Textbox(label='Supervisor Name')
                        sup_e   = gr.Textbox(label='Supervisor Email')
                        btn     = gr.Button('Run Safety Inspection',
                                            variant='primary', elem_id='inspect-btn')
                        gr.Examples([['test_videos/no_hardhat.mp4','Site A','John M.','j@x.com'],
                                     ['test_videos/fall_risk.mp4', 'Site B','Sara C.','s@x.com']],
                                    inputs=[media,site,sup_n,sup_e])
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.TabItem('Violations'):
                                viol_out  = gr.HTML()
                                frame_out = gr.Image(label='Sample Frame',height=240)
                            with gr.TabItem('OSHA Report'):
                                rep_out   = gr.Textbox(lines=25,show_copy_button=True)
                            with gr.TabItem('Supervisor Email'):
                                subj_out  = gr.Textbox(label='Subject',show_copy_button=True)
                                body_out  = gr.Textbox(lines=18,show_copy_button=True)
                btn.click(run_ui,[media,site,sup_n,sup_e],
                          [viol_out,rep_out,subj_out,body_out,frame_out],show_progress=True)
            with gr.TabItem('Dashboard'):
                stats = get_dashboard_stats()
                gr.Metric('Total Inspections',stats['total_inspections'])
                gr.Metric('Total Violations',stats['total_violations'])
                gr.Metric('High/Critical Events',stats['high_risk_inspections'])
    return demo

if __name__ == '__main__':
    build_ui().launch(server_name='0.0.0.0', server_port=7860)
