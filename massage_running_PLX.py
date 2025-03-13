import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from massage_macroPLX_f1 import massage_macroPLX

input_file=sys.argv[1]
config_file=sys.argv[2]

conf1 = pd.read_csv(config_file, sep=',')
print("RUNNING MASSAGE WITH:")
print(conf1)
parameter = conf1['parameter'].to_numpy()
value = conf1['option'].to_numpy()
Xopt1=value[11]
output1=value[14]

dat1 = pd.read_csv(input_file, sep=',')
ID1= np.array(dat1[['Source']])
RA=np.array(dat1[['RAJ2000_1']])
DEC=np.array(dat1[['DEJ2000_1']])
Bp=np.array(dat1[['BPmag']])
Gp=np.array(dat1[['Gmag']])
Rp=np.array(dat1[['RPmag']])
J=np.array(dat1[['Jmag']])
H=np.array(dat1[['Hmag']])
eBp=np.array(dat1[['e_BPmag']])
eGp=np.array(dat1[['e_Gmag']])
eRp=np.array(dat1[['e_RPmag']])
eJ=np.array(dat1[['e_Jmag']])
eH=np.array(dat1[['e_Hmag']])
if Xopt1 == "Teff":
  XX=np.array(dat1[['Teff']])
  eXX=eTeff=np.array(dat1[['eTeff']])
if Xopt1 == "lgTeff":
  XX=np.array(dat1[['lgTeff']])
  eXX=eTeff=np.array(dat1[['elgTeff']])
if Xopt1 == "SpT_PM13" or Xopt1 == "SpT_KH95":
  XX=np.array(dat1[['SpT']])
  eXX=np.array(dat1[['SpTerr']])
PLX=np.array(dat1[['PLXcorr']])
ePLX=np.array(dat1[['e_Plx']])
nobj=PLX.size
nobj=3

outdef= np.empty((nobj,51), dtype=np.float32)
outputfile2=output1 + "_" + "results" + ".csv"

fmt1= '%25s,%10.6f,%10.6f,%7.3f,%7.3f,%7.3f,%7.3f,%7.3f,%7.3f,%7.3f,%7.3f,%5.2f,%1d,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%8.1f,%5.2f,%8.3f,%8.3f,%8.3f,%8.3f,%8.3f,%8.3f,%5.2f,%8.3f,%8.3f,%8.3f,%8.3f,%8.3f,%8.3f,%5.2f,%8.3f,%8.3f,%8.3f,%8.3f,%8.3f,%8.3f,%5.2f,%8.3f,%8.3f,%8.3f,%8.3f,%8.3f,%8.3f,%5.2f,%5d,%5.2f,%2d'
head1='OBJECT,RA,DEC,Av_wmean,eAv_wmean,Av_mean,Av_std,Av_median,Av_mad,Av_low,Av_upp,Av_skew,nfilter_1,T_mean,T_std,T_median,T_mad,T_low,T_upp,T_skew,lgL_mean,lgL_std, lgL_median,lgL_mad,lgL_low,lgL_upp,lgL_skew, mass_mean,mass_std,mass_median,mass_mad,mass_low,mass_upp,mass_skew,age_mean,age_std, age_median,age_mad,age_low,age_upp,age_skew, lgg_mean,lgg_std, lgg_median,lgg_mad,lgg_low,lgg_upp,lgg_skew,nmc1,perc_num,qflag'

for i1 in range(nobj):
  staT = time.time()
  print(ID1[i1,0])
  outdef[i1,]=massage_macroPLX(ID1[i1,0], RA[i1,0],DEC[i1,0],XX[i1,0], eXX[i1,0], Bp[i1,0], eBp[i1,0], Gp[i1,0], eGp[i1,0], Rp[i1,0], eRp[i1,0],J[i1,0], eJ[i1,0], H[i1,0], eH[i1,0], PLX[i1,0], ePLX[i1,0], "massage_config.txt")
  endT = time.time()
  print(endT-staT, "seconds")
np.savetxt(outputfile2, outdef, delimiter=',', fmt=fmt1, header=head1, comments='')
  
