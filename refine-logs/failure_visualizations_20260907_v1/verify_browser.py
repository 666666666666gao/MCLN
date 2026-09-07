import json
from pathlib import Path
from playwright.sync_api import sync_playwright

root=Path('C:/Users/gb/Desktop/document/MCLN_3D_failure_visualizations_20260907')
manifest=json.loads((root/'cases.json').read_text(encoding='utf-8'))
results=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True,args=['--allow-file-access-from-files','--enable-unsafe-swiftshader'])
    context=browser.new_context(viewport={'width':1440,'height':1200},offline=True)
    page=context.new_page()
    errors=[]
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto((root/'index.html').as_uri())
    assert page.locator('article').count()==6
    assert page.locator('img').evaluate_all('(images)=>images.every(im=>im.complete&&im.naturalWidth>0)')
    for i,c in enumerate(manifest['cases']):
        page.goto((root/c['rendering']['interactive_html']).as_uri(),wait_until='load')
        page.wait_for_function("document.querySelector('.js-plotly-plot') && document.querySelector('.js-plotly-plot')._fullLayout && document.querySelector('canvas')",timeout=30000)
        values=page.evaluate("""() => {
            const plot=document.querySelector('.js-plotly-plot');
            return {traces:plot.data.length,points:plot.data[0].x.length,gtPoints:plot.data[3].x.length,canvasCount:document.querySelectorAll('canvas').length,aspectMode:plot.layout.scene.aspectmode};
        }""")
        assert values['traces']==4 and values['points']==50000 and values['gtPoints']==c['target_points'] and values['aspectMode']=='data'
        assert not errors,errors
        if i==0:
            page.screenshot(path=str(root/'interactive_preview.png'),full_page=True)
        results.append({'case_id':c['case_id'],'offline_browser_loaded':True,**values})
        print('BROWSER_PASS',c['case_id'],flush=True)
    browser.close()
(root/'browser_verification.json').write_text(json.dumps({'engine':'installed Chrome, headless','network':'offline','pages':results,'page_errors':errors},indent=2)+'\n',encoding='utf-8')
