"""Task 2: Image Inpainting & Editing — custom canvas mask tools."""

import base64
import io

import gradio as gr
import numpy as np
from PIL import Image


MAX_T2_LONG = 3840


def _img_to_data_url(img):
    if img is None:
        return ""
    if max(img.width, img.height) > 1024:
        s = 1024 / max(img.width, img.height)
        img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return "data:image/jpeg;base64," + b64


def _mask_from_data_url(b64_str, ref_img):
    if not b64_str or ref_img is None:
        return None
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        raw = base64.b64decode(b64_str)
        mask_img = Image.open(io.BytesIO(raw)).convert("RGBA")
        arr = np.array(mask_img)
        alpha = arr[:, :, 3]
        mask_arr = np.where(alpha > 10, 255, 0).astype(np.uint8)
        mask_pil = Image.fromarray(mask_arr, mode="L")
        if (mask_pil.width, mask_pil.height) != (ref_img.width, ref_img.height):
            mask_pil = mask_pil.resize((ref_img.width, ref_img.height), Image.NEAREST)
        return mask_pil
    except Exception:
        return None


def _not_implemented(input_img, mask_b64, prompt, step, task_type):
    if input_img is None:
        raise gr.Error("Vui long upload anh truoc.")
    if max(input_img.width, input_img.height) > MAX_T2_LONG:
        raise gr.Error(f"Anh qua lon - toi da 4K ({MAX_T2_LONG}px canh dai).")
    raise gr.Error("Task 2 model not yet implemented.")


_CANVAS_HTML = """
<style>
.t2c-hidden{display:none!important}
#t2c-wrap{background:#1e1e1e;border-radius:8px;overflow:hidden;font-family:sans-serif;user-select:none}
#t2c-toolbar{display:flex;gap:5px;padding:7px 10px;flex-wrap:wrap;background:#2d2d2d;align-items:center;border-bottom:1px solid #444}
.t2c-btn{padding:4px 9px;border-radius:5px;border:1px solid #555;background:#3a3a3a;color:#eee;cursor:pointer;font-size:13px;transition:background .15s}
.t2c-btn:hover{background:#4a4a4a}
.t2c-btn.active{background:#0066cc;border-color:#0088ff;color:#fff;font-weight:bold}
.t2c-btn.danger{background:#7a2a2a;border-color:#993333}
.t2c-btn.danger:hover{background:#993333}
#t2c-stage{position:relative;overflow:auto;background:#2a2a2a;display:flex;justify-content:center;align-items:flex-start;min-height:220px;max-height:460px}
#t2c-bg-canvas,#t2c-mask-canvas{display:block;position:absolute;top:0;left:0}
#t2c-bg-canvas{pointer-events:none}
#t2c-mask-canvas{cursor:crosshair}
#t2c-ph{color:#777;font-size:13px;padding:90px 30px;text-align:center;pointer-events:none;position:relative;z-index:0}
#t2c-hint{padding:3px 10px;font-size:11px;color:#aaa;background:#232323;min-height:18px}
.t2c-sep{width:1px;height:20px;background:#555;margin:0 3px}
</style>
<div id="t2c-wrap">
  <div id="t2c-toolbar">
    <button class="t2c-btn active" id="t2c-btn-brush"   onclick="t2cTool('brush')">&#x270F; Brush</button>
    <button class="t2c-btn"        id="t2c-btn-eraser"  onclick="t2cTool('eraser')">&#x2715; Eraser</button>
    <div class="t2c-sep"></div>
    <button class="t2c-btn"        id="t2c-btn-rect"    onclick="t2cTool('rect')">&#x25A1; Rectangle</button>
    <button class="t2c-btn"        id="t2c-btn-circle"  onclick="t2cTool('circle')">&#x25CB; Ellipse</button>
    <button class="t2c-btn"        id="t2c-btn-polygon" onclick="t2cTool('polygon')">&#x2B21; Polygon</button>
    <div class="t2c-sep"></div>
    <span style="color:#999;font-size:12px">Size:</span>
    <input id="t2c-sz" type="range" min="2" max="120" value="25" style="width:70px;accent-color:#0088ff" oninput="document.getElementById('t2c-szv').textContent=this.value;t2cSetSz(+this.value)">
    <span id="t2c-szv" style="color:#bbb;font-size:11px;min-width:22px">25</span>
    <div class="t2c-sep"></div>
    <button class="t2c-btn" onclick="t2cUndo()" title="Undo">&#x21A9; Undo</button>
    <button class="t2c-btn danger" onclick="t2cClear()" title="Clear mask">&#x1F5D1; Clear</button>
  </div>
  <div id="t2c-stage">
    <div id="t2c-ph">&#x2B06; Upload image above<br>then paint the area to edit</div>
    <canvas id="t2c-bg-canvas"></canvas>
    <canvas id="t2c-mask-canvas"></canvas>
  </div>
  <div id="t2c-hint"></div>
</div>
<script>
(function(){
'use strict';
var COLOR='rgba(255,68,68,0.35)';
var bgC=document.getElementById('t2c-bg-canvas');
var mC=document.getElementById('t2c-mask-canvas');
var hint=document.getElementById('t2c-hint');
var ph=document.getElementById('t2c-ph');
var bgX=bgC.getContext('2d'),mX=mC.getContext('2d');
var tool='brush',sz=25,drawing=false,sx=0,sy=0;
var polyPts=[],history=[],preSnap=null,loaded=false,lastSrc='';
window.t2cTool=function(t){
  tool=t;polyPts=[];
  document.querySelectorAll('.t2c-btn').forEach(function(b){b.classList.remove('active');});
  var btn=document.getElementById('t2c-btn-'+t);
  if(btn)btn.classList.add('active');
  hint.textContent=t==='polygon'?'Click to add anchor points, double-click to close and fill':''
;};
window.t2cSetSz=function(v){sz=v;};
window.t2cClear=function(){mX.clearRect(0,0,mC.width,mC.height);history=[];polyPts=[];push();};
window.t2cUndo=function(){
  if(history.length===0)return;
  history.pop();
  if(history.length)mX.putImageData(history[history.length-1],0,0);
  else mX.clearRect(0,0,mC.width,mC.height);
  push();
};
window.t2cLoad=function(src){
  if(!src||src===lastSrc)return;lastSrc=src;
  var img=new Image();img.crossOrigin='anonymous';
  img.onload=function(){
    var maxH=440;
    var scale=img.naturalHeight>maxH?maxH/img.naturalHeight:1;
    var dw=Math.round(img.naturalWidth*scale),dh=Math.round(img.naturalHeight*scale);
    bgC.width=mC.width=dw;bgC.height=mC.height=dh;
    bgC.style.width=mC.style.width=dw+'px';
    bgC.style.height=mC.style.height=dh+'px';
    document.getElementById('t2c-stage').style.minHeight=dh+'px';
    bgX.drawImage(img,0,0,dw,dh);mX.clearRect(0,0,dw,dh);
    history=[];loaded=true;ph.style.display='none';push();
  };
  img.src=src;
};
function snap(){history.push(mX.getImageData(0,0,mC.width,mC.height));if(history.length>30)history.shift();}
function pos(e){
  var r=mC.getBoundingClientRect();
  var s=e.touches?e.touches[0]:e;
  return{x:(s.clientX-r.left)*(mC.width/r.width),y:(s.clientY-r.top)*(mC.height/r.height)};
}
mC.addEventListener('mousedown',function(e){e.preventDefault();down(e);});
mC.addEventListener('mousemove',function(e){e.preventDefault();move(e);});
mC.addEventListener('mouseup',function(e){e.preventDefault();up(e);});
mC.addEventListener('mouseleave',function(e){if(drawing)up(e);});
mC.addEventListener('dblclick',function(e){e.preventDefault();dbl(e);});
mC.addEventListener('touchstart',function(e){e.preventDefault();down(e);},{passive:false});
mC.addEventListener('touchmove',function(e){e.preventDefault();move(e);},{passive:false});
mC.addEventListener('touchend',function(e){e.preventDefault();up(e);},{passive:false});
function down(e){
  if(loaded===false)return;
  var p=pos(e);
  if(tool==='polygon'){addPoly(p);return;}
  drawing=true;sx=p.x;sy=p.y;
  if(tool==='brush'||tool==='eraser'){snap();mX.beginPath();mX.moveTo(p.x,p.y);}
  else{preSnap=mX.getImageData(0,0,mC.width,mC.height);}
}
function move(e){
  if(drawing===false)return;
  var p=pos(e);
  if(tool==='brush'){
    mX.globalCompositeOperation='source-over';
    mX.strokeStyle=COLOR;mX.lineWidth=sz;mX.lineCap=mX.lineJoin='round';
    mX.lineTo(p.x,p.y);mX.stroke();mX.beginPath();mX.moveTo(p.x,p.y);
  }else if(tool==='eraser'){
    mX.globalCompositeOperation='destination-out';
    mX.lineWidth=sz;mX.lineCap=mX.lineJoin='round';
    mX.lineTo(p.x,p.y);mX.stroke();mX.beginPath();mX.moveTo(p.x,p.y);
    mX.globalCompositeOperation='source-over';
  }else if(tool==='rect'){
    mX.putImageData(preSnap,0,0);
    mX.globalCompositeOperation='source-over';
    mX.fillStyle=COLOR;mX.fillRect(sx,sy,p.x-sx,p.y-sy);
  }else if(tool==='circle'){
    mX.putImageData(preSnap,0,0);
    var rx=Math.abs(p.x-sx)/2,ry=Math.abs(p.y-sy)/2;
    var cx=(sx+p.x)/2,cy=(sy+p.y)/2;
    mX.globalCompositeOperation='source-over';
    mX.fillStyle=COLOR;mX.beginPath();
    mX.ellipse(cx,cy,Math.max(rx,1),Math.max(ry,1),0,0,2*Math.PI);mX.fill();
  }
}
function up(e){
  if(drawing===false)return;drawing=false;
  if(tool==='rect'||tool==='circle'){snap();preSnap=null;}
  mX.globalCompositeOperation='source-over';
  push();
}
function addPoly(p){
  polyPts.push(p);
  mX.fillStyle='rgba(255,220,0,0.9)';
  mX.beginPath();mX.arc(p.x,p.y,3,0,2*Math.PI);mX.fill();
  if(polyPts.length>=2){
    var pr=polyPts[polyPts.length-2];
    mX.strokeStyle='rgba(255,220,0,0.7)';mX.lineWidth=1.5;
    mX.beginPath();mX.moveTo(pr.x,pr.y);mX.lineTo(p.x,p.y);mX.stroke();
  }
  hint.textContent=polyPts.length+' points - double-click to close and fill';
}
function dbl(e){
  if(tool!=='polygon'||polyPts.length<3)return;
  if(history.length)mX.putImageData(history[history.length-1],0,0);
  else mX.clearRect(0,0,mC.width,mC.height);
  snap();
  mX.globalCompositeOperation='source-over';
  mX.fillStyle=COLOR;mX.beginPath();
  mX.moveTo(polyPts[0].x,polyPts[0].y);
  for(var i=1;i<polyPts.length;i++)mX.lineTo(polyPts[i].x,polyPts[i].y);
  mX.closePath();mX.fill();
  polyPts=[];hint.textContent='Click to add anchor points, double-click to close and fill';
  push();
}
function push(){
  var b64=mC.toDataURL('image/png');
  function tryPush(){
    var el=document.querySelector('#t2c-mask-out textarea');
    if(el===null){setTimeout(tryPush,300);return;}
    el.value=b64;el.dispatchEvent(new InputEvent('input',{bubbles:true}));
  }
  tryPush();
}
setInterval(function(){
  var el=document.querySelector('#t2c-bg-url textarea');
  if(el===null)return;
  if(el.value&&el.value!==lastSrc)t2cLoad(el.value);
},400);
})();
</script>
"""


def create_task2_tab():
    gr.Markdown(
        "**Image Inpainting & Editing** — Upload an image, paint the area you want to edit "
        "using the tools below, then describe the edit and click **Run**."
    )

    with gr.Row():
        task_type = gr.Radio(
            choices=["Replace", "Delete", "Add"],
            value="Replace",
            label="Task Type",
            scale=1,
        )
        step = gr.Number(label="Step", value=1, minimum=1, maximum=50, precision=0, scale=1)
        gr.HTML("<div></div>", scale=4)

    with gr.Row():
        # Left: input image + drawing canvas
        with gr.Column(scale=1):
            gr.Markdown("### Canvas \u2014 draw your mask")
            input_image = gr.Image(
                type="pil",
                label="Input Image + Mask",
                sources=["upload", "clipboard"],
            )
            gr.HTML(_CANVAS_HTML)
            bg_url_tb   = gr.Textbox(visible=True, elem_id="t2c-bg-url",   elem_classes=["t2c-hidden"], label="", container=False)
            mask_out_tb = gr.Textbox(visible=True, elem_id="t2c-mask-out", elem_classes=["t2c-hidden"], label="", container=False)

        # Middle: mask preview
        with gr.Column(scale=1):
            gr.Markdown("### Mask Preview")
            binary_mask = gr.Image(
                label="Binary Mask (white = edit area)",
                type="pil",
                interactive=False,
                height=480,
            )

        # Right: output
        with gr.Column(scale=1):
            gr.Markdown("### Output")
            output_image = gr.Image(
                label="Result",
                type="pil",
                interactive=False,
                height=480,
                sources=[],
            )

    with gr.Row():
        prompt = gr.Textbox(
            label="Prompt",
            placeholder="Describe the edit (e.g. 'replace with a wooden table')",
            scale=4,
        )
        run_btn = gr.Button("Run", variant="primary", scale=1, size="lg")

    input_image.change(
        fn=_img_to_data_url,
        inputs=[input_image],
        outputs=[bg_url_tb],
        queue=False,
    ).then(
        fn=None,
        inputs=[bg_url_tb],
        outputs=[],
        js="(url) => { if(url && window.t2cLoad) window.t2cLoad(url); }",
        queue=False,
    )

    mask_out_tb.change(
        fn=_mask_from_data_url,
        inputs=[mask_out_tb, input_image],
        outputs=binary_mask,
        queue=False,
    )

    run_btn.click(
        fn=_not_implemented,
        inputs=[input_image, mask_out_tb, prompt, step, task_type],
        outputs=output_image,
    )
    prompt.submit(
        fn=_not_implemented,
        inputs=[input_image, mask_out_tb, prompt, step, task_type],
        outputs=output_image,
    )
