#!/usr/bin/env python3
"""Probe public research-return archives for fixed global factor study; no trading."""
import io,zipfile,requests
FILES={
"us_value":"6_Portfolios_2x3_CSV.zip",
"us_quality":"6_Portfolios_ME_OP_2x3_CSV.zip",
"us_momentum":"6_Portfolios_ME_Prior_12_2_CSV.zip",
"us_market":"F-F_Research_Data_5_Factors_2x3_CSV.zip",
"developed_ex_us_value":"Developed_ex_US_6_Portfolios_ME_BE-ME_CSV.zip",
"developed_ex_us_quality":"Developed_ex_US_6_Portfolios_ME_OP_CSV.zip",
"developed_ex_us_momentum":"Developed_ex_US_6_Portfolios_ME_Prior_12_2_CSV.zip",
"developed_ex_us_market":"Developed_ex_US_5_Factors_CSV.zip",
"emerging_value":"Emerging_Markets_6_Portfolios_ME_BE-ME_CSV.zip",
"emerging_quality":"Emerging_Markets_6_Portfolios_ME_OP_CSV.zip",
"emerging_momentum":"Emerging_Markets_6_Portfolios_ME_Prior_12_2_CSV.zip",
"emerging_market":"Emerging_5_Factors_CSV.zip",
}
BASE="https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
for label,filename in FILES.items():
    try:
        r=requests.get(BASE+filename,timeout=55,headers={"User-Agent":"GlobalAlphaPersonalResearch/0.27"})
        r.raise_for_status()
        if not r.content.startswith(b"PK"):raise ValueError("Not a ZIP file")
        z=zipfile.ZipFile(io.BytesIO(r.content))
        f=z.namelist()[0]
        text=z.read(f).decode("latin1")
        lines=text.splitlines()
        starts=[(i,line[:185]) for i,line in enumerate(lines) if line.strip().startswith(("199001","200001","202509"))]
        print("FRENCH_DATA",label,"ARCHIVE",filename,"SIZE",len(r.content),"HEAD",repr(lines[:15]),"EARLY",starts[:3],"END",starts[-2:],flush=True)
    except Exception as e:
        print("FRENCH_ERROR",label,type(e).__name__,str(e)[:150],flush=True)
