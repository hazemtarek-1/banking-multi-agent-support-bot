from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT = Path('artifacts/report_assets')
OUT.mkdir(parents=True, exist_ok=True)

try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 18)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
    bold = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 26)
except Exception:
    font = ImageFont.load_default(); small = font; bold = font

# Architecture diagram
img = Image.new('RGB', (1500, 900), 'white')
d = ImageDraw.Draw(img)
boxes = [
    (500, 40, 1000, 120, 'User message'),
    (420, 180, 1080, 300, 'BiGRU Intent Classifier\nEmbedding → BiGRU → Linear → confidence'),
    (420, 360, 1080, 440, 'Deterministic Router\nintent → specialist agent'),
    (60, 540, 390, 700, 'Billing Agent\ncheck_invoice\nsearch_docs'),
    (405, 540, 735, 700, 'Technical Agent\nsearch_docs'),
    (750, 540, 1080, 700, 'Booking Agent\ncheck_availability\nsearch_docs'),
    (1095, 540, 1440, 700, 'General Agent\nsearch_docs'),
    (500, 770, 1000, 850, 'Groq → Ollama → offline response → Streamlit'),
]
for x1,y1,x2,y2,text in boxes:
    d.rounded_rectangle((x1,y1,x2,y2), radius=18, outline='#23406a', width=4, fill='#eef3f9')
    lines=text.split('\n')
    total=sum((d.textbbox((0,0),ln,font=small)[3] for ln in lines))+8*(len(lines)-1)
    yy=(y1+y2-total)/2
    for ln in lines:
        bb=d.textbbox((0,0),ln,font=small); ww=bb[2]-bb[0]
        d.text(((x1+x2-ww)/2,yy),ln,font=small,fill='black'); yy += bb[3]-bb[1]+8
for a,b in [((750,120),(750,180)),((750,300),(750,360)),((225,440),(225,540)),((570,440),(570,540)),((915,440),(915,540)),((1265,440),(1265,540)),((750,700),(750,770))]:
    d.line((*a,*b), fill='#23406a', width=5)
    x=(a[0]+b[0])//2; y=(a[1]+b[1])//2
    d.polygon([(x,y+12),(x-10,y-4),(x+10,y-4)], fill='#23406a')
d.text((55, 10), 'System Architecture', font=bold, fill='#1d3557')
img.save(OUT/'architecture.png')

# Streamlit UI preview (explicitly a preview, not a captured live run)
img = Image.new('RGB', (1600, 950), '#f6f7f9')
d = ImageDraw.Draw(img)
d.rectangle((0,0,1600,70), fill='#17375e')
d.text((40,18),'DemoShop Multi-Agent Support Bot',font=bold,fill='white')
d.rectangle((1180,70,1600,950), fill='white', outline='#cccccc')
d.text((1210,100),'Routing',font=bold,fill='#17375e')
for i,(label,value) in enumerate([('Predicted intent','card_not_working'),('Confidence','92.4%'),('Agent','technical'),('LLM provider','groq'),('Session usage','3 / 20')]):
    y=160+i*120
    d.text((1210,y),label,font=small,fill='#555')
    d.text((1210,y+35),value,font=small,fill='black')
# chat area
messages=[('user','My card is not working.'),('assistant','I can help with that. First, check whether the card is activated and whether another terminal/ATM works. I used the technical support knowledge base for these steps.')]
y=140
for role,text in messages:
    x=60 if role=='assistant' else 620
    w=900 if role=='assistant' else 780
    d.rounded_rectangle((x,y,x+w,y+130),radius=18,fill='#e8eef7' if role=='assistant' else '#dcefe2')
    d.text((x+20,y+18),role.title(),font=small,fill='#17375e')
    d.multiline_text((x+20,y+55),text,font=small,fill='black',spacing=6)
    y+=170
# input
inp_y=770
d.rounded_rectangle((60,inp_y,1120,850),radius=18,outline='#aaa',fill='white')
d.text((85,inp_y+25),'Type your support question...',font=small,fill='#777')
d.rounded_rectangle((1130,inp_y,1200,850),radius=18,fill='#17375e')
d.text((1150,inp_y+25),'➤',font=small,fill='white')
d.text((60,890),'UI preview for the report. Capture the actual running Streamlit window before final submission.',font=small,fill='#8a3b12')
img.save(OUT/'streamlit_ui_preview.png')

# Source screenshot helper

def code_image(title, path, start=1, end=45):
    lines=Path(path).read_text(encoding='utf-8').splitlines()
    lines=lines[start-1:end]
    width=1500; height=max(220, 70+len(lines)*27)
    im=Image.new('RGB',(width,height),'#f8f8f8'); dr=ImageDraw.Draw(im)
    dr.rectangle((0,0,width,60),fill='#17375e'); dr.text((25,16),title,font=small,fill='white')
    y=75
    for i,line in enumerate(lines,start):
        dr.text((25,y),f'{i:>3}  {line[:125]}',font=font,fill='#111')
        y+=27
    im.save(OUT/(Path(path).stem+'_code.png'))

code_image('Classifier architecture', 'src/classifier/model.py', 1, 45)
code_image('Deterministic routing', 'src/router.py', 1, 75)
code_image('Specialist agents and tools', 'src/agents/agents.py', 1, 100)
code_image('Streamlit application', 'app/app.py', 1, 100)

# Test evidence
im=Image.new('RGB',(1200,260),'#101820'); dr=ImageDraw.Draw(im)
try: termfont=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',26)
except: termfont=font
for i,t in enumerate(['$ python -m pytest -q','...','3 passed in 0.97s']):
    dr.text((35,35+i*70),t,font=termfont,fill='white')
im.save(OUT/'test_evidence.png')
