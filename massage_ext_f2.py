# A function version of the modulus to obtain extinctions from GAIADR3-2MASS photometry and Teff or SpT
import pandas as pd
import numpy as np
import sys
import os
import random
#import time
import math
#import matplotlib.pyplot as plt
#import seaborn as sns
from scipy import interpolate
from extinction import al_ext
#massagedir='/home/hernandj/ASTROTOOLS/IDLpro/JHpro/MASSAGE_V2/'

#staT = time.time()
def massage_ext2 (Teff, eTeff,Xopt1, Bp, Bperr, Gp, Gperr, Rp, Rperr, Jmag, Jerr, Hmag, Herr, Mnorm, Yopt1, nmc1, Rv1, elaw1, PLX, PLXerr,inter_mode,massagedir):
  kind1=inter_mode
  if Yopt1 == "KH95" or Yopt1 == "PM13":
    # READING STANDARD TABLES
    # POR LOS MOMENTOS PECAUT & MAMAJEK 2013, LUHMAN & ESPLIN 2020, KENYON & HARTMANN 1995 
    tmp1='MODELS/EsplinLuhman2020_KH95_PM13.cat_GAIAEDR3.csv'
    file1 = os.path.join(massagedir, tmp1)
    std1 = pd.read_csv(file1, sep=',')
    # VALORES ESTANDARES
    SPT0 = np.array(std1[['SPT']])
    Teffp0 = np.array(std1[['Teffp']])
    Teffk0 = np.array(std1[['Teffk']])
    BpRp0 = np.array(std1[['BpRpEDR3']])
    GpRp0 = np.array(std1[['GpRpEDR3']])
    RpJ0 = np.array(std1[['RpJEDR3']])
    JH0 = np.array(std1[['JH']])
    VJk0 = np.array(std1[['VJk']])
    BCVk0 = np.array(std1[['BCVk']])
    BCJk0 = VJk0+BCVk0
    BCRk0 = BCJk0+RpJ0  
    BCGk0 = BCRk0+GpRp0
    BCVp0 = np.array(std1[['BCVp']])
    BCJp0 = np.array(std1[['BCJp']])
    BCRp0 = BCJp0+RpJ0
    BCGp0 = BCRp0+GpRp0
 
    #FUNCIONES DE INTERPOLACION 
    f_TeffPM13 = interpolate.interp1d(SPT0.flatten(), Teffp0.flatten(), kind=kind1,fill_value="extrapolate")
    f_TeffKH95 = interpolate.interp1d(SPT0.flatten(), Teffk0.flatten(), kind=kind1,fill_value="extrapolate")
    f_JH = interpolate.interp1d(Teffp0.flatten(), JH0.flatten(), kind=kind1,fill_value="extrapolate")
    f_BpRp = interpolate.interp1d(Teffp0.flatten(), BpRp0.flatten(), kind=kind1,fill_value="extrapolate")
    f_GpRp = interpolate.interp1d(Teffp0.flatten(), GpRp0.flatten(), kind=kind1,fill_value="extrapolate")
    f_RpJ = interpolate.interp1d(Teffp0.flatten(), RpJ0.flatten(), kind=kind1,fill_value="extrapolate")
    if Yopt1 == "KH95":
      f_BCJ = interpolate.interp1d(Teffp0.flatten(), BCJk0.flatten(), kind=kind1,fill_value="extrapolate")
      f_BCRp = interpolate.interp1d(Teffp0.flatten(), BCRk0.flatten(), kind=kind1,fill_value="extrapolate")
      f_BCGp = interpolate.interp1d(Teffp0.flatten(), BCGk0.flatten(), kind=kind1,fill_value="extrapolate")
    if Yopt1 == "PM13":
      f_BCJ = interpolate.interp1d(Teffp0.flatten(), BCJp0.flatten(), kind=kind1,fill_value="extrapolate")
      f_BCRp = interpolate.interp1d(Teffp0.flatten(), BCRp0.flatten(), kind=kind1,fill_value="extrapolate")
      f_BCGp = interpolate.interp1d(Teffp0.flatten(), BCGp0.flatten(), kind=kind1,fill_value="extrapolate")

  if Yopt1 == "MIST":
    tmp1='MODELS/Colors_MIST_1to100Myr_lt20Msun.csv'
    file1 = os.path.join(massagedir, tmp1)
    std1 = pd.read_csv(file1, sep=',')
    # VALORES ESTANDARES
    Teff0 = np.array(std1[['Teff']])
    BpGp0 = np.array(std1[['Bp_Gp']])
    GpRp0 = np.array(std1[['Gp_Rp']])
    GpJ0 = np.array(std1[['Gp_J']])
    GpH0 = np.array(std1[['Gp_H']])
    BpRp0 = BpGp0+GpRp0
    RpJ0 = GpJ0-GpRp0
    JH0 = GpH0-GpJ0
    BCV0 = np.array(std1[['BCv']])
    BCJ0 = np.array(std1[['BCj']])
    BCRp0 = BCJ0+RpJ0
    BCGp0 = BCRp0+GpRp0
    #FUNCIONES DE INTERPOLACION 
    f_JH = interpolate.interp1d(Teff0.flatten(), JH0.flatten(), kind=kind1,fill_value="extrapolate")
    f_BpRp = interpolate.interp1d(Teff0.flatten(), BpRp0.flatten(), kind=kind1,fill_value="extrapolate")
    f_GpRp = interpolate.interp1d(Teff0.flatten(), GpRp0.flatten(), kind=kind1,fill_value="extrapolate")
    f_RpJ = interpolate.interp1d(Teff0.flatten(), RpJ0.flatten(), kind=kind1,fill_value="extrapolate")
    f_BCJ = interpolate.interp1d(Teff0.flatten(), BCJ0.flatten(), kind=kind1,fill_value="extrapolate")
    f_BCRp = interpolate.interp1d(Teff0.flatten(), BCRp0.flatten(), kind=kind1,fill_value="extrapolate")
    f_BCGp = interpolate.interp1d(Teff0.flatten(), BCGp0.flatten(), kind=kind1,fill_value="extrapolate")

  if Yopt1 == "PARSEC":
    tmp1='MODELS/Colors_PARSEC_1to100Myr_lt20Msun.csv'
    file1 = os.path.join(massagedir, tmp1)
    std1 = pd.read_csv(file1, sep=',')
    # VALORES ESTANDARES
    Teff0 = np.array(std1[['Teff']])
    BpGp0 = np.array(std1[['Bp_Gp']])
    GpRp0 = np.array(std1[['Gp_Rp']])
    GpJ0 = np.array(std1[['Gp_J']])
    GpH0 = np.array(std1[['Gp_H']])
    BpRp0 = BpGp0+GpRp0
    RpJ0 = GpJ0-GpRp0
    JH0 = GpH0-GpJ0
    BCV0 = np.array(std1[['BCv']])
    BCJ0 = np.array(std1[['BCj']])
    BCRp0 = BCJ0+RpJ0
    BCGp0 = BCRp0+GpRp0
    #FUNCIONES DE INTERPOLACION 
    f_JH = interpolate.interp1d(Teff0.flatten(), JH0.flatten(), kind=kind1,fill_value="extrapolate")
    f_BpRp = interpolate.interp1d(Teff0.flatten(), BpRp0.flatten(), kind=kind1,fill_value="extrapolate")
    f_GpRp = interpolate.interp1d(Teff0.flatten(), GpRp0.flatten(), kind=kind1,fill_value="extrapolate")
    f_RpJ = interpolate.interp1d(Teff0.flatten(), RpJ0.flatten(), kind=kind1,fill_value="extrapolate")
    f_BCJ = interpolate.interp1d(Teff0.flatten(), BCJ0.flatten(), kind=kind1,fill_value="extrapolate")
    f_BCRp = interpolate.interp1d(Teff0.flatten(), BCRp0.flatten(), kind=kind1,fill_value="extrapolate")
    f_BCGp = interpolate.interp1d(Teff0.flatten(), BCGp0.flatten(), kind=kind1,fill_value="extrapolate")

# AV ARRAY, From Av=-1 to Av=20 
  Av = (np.arange(2500)/ 100.0)-5.0
  Diff = np.empty(2500, dtype=np.float32)
  # VALORES DE LOS FILTROS, PUEDEN CAMBIAR CON LAS FUNCIONES AVANZADAS
  wl1=np.array([0.518,0.639,0.783,1.238,1.648])
  Al1=np.empty(5, dtype=np.float32)
  for a1 in range(5):
     Al1[a1]=al_ext(wl1[a1],Rv1,elaw1,massagedir)
#  Al1=np.array([1.083,0.836,0.634,0.286,0.181])
  # SELECCIONA VALOES BUENOS
  eMtmp1=np.array([Bperr,Gperr,Rperr,Jerr,Herr])
  errMlim=1.0 #error máximo
  sel1=np.array(np.where((eMtmp1<=errMlim)))
  #print(sel1.shape)

  # RANDOM VALUES GENERATION
  if Xopt1 == "Teff":
     T1=np.random.normal(Teff, eTeff,nmc1)
  if Xopt1 == "lgTeff":
     lgT1=np.random.normal(Teff, eTeff,nmc1)
     T1=10**lgT1
  if Xopt1 == "SpT_PM13":
     S1=np.random.normal(Teff, eTeff,nmc1)
     T1=f_TeffPM13(S1)
  if Xopt1 == "SpT_KH95":
     S1=np.random.normal(Teff, eTeff,nmc1)
     T1=f_TeffKH95(S1)
  Bp1=np.random.normal(Bp,Bperr,nmc1)
  Gp1=np.random.normal(Gp,Gperr,nmc1)
  Rp1=np.random.normal(Rp,Rperr,nmc1)
  J1=np.random.normal(Jmag,Jerr,nmc1)
  H1=np.random.normal(Hmag,Herr,nmc1)
  PLX1=np.random.normal(PLX,PLXerr,nmc1)
  Avout = np.empty(nmc1, dtype=np.float32)
  Dout= np.empty(nmc1, dtype=np.float32)
  lgLout= np.empty(nmc1, dtype=np.float32)
  BC1=np.empty(nmc1, dtype=np.float32)
  npt1=np.full(nmc1,sel1.size)
  #GENERACION MONTECARLO
  if Mnorm == "J" and Jerr<=errMlim:
    All1=Al1[3]
    Mobs1=J1
    for i1 in range(nmc1):
      BC1[i1]= f_BCJ(T1[i1])
      JH_phot = f_JH(T1[i1])
      BpRp_phot = f_BpRp(T1[i1])
      GpRp_phot = f_GpRp(T1[i1])
      RpJ_phot = f_RpJ(T1[i1])
      Mag1=np.array([Bp1[i1],Gp1[i1],Rp1[i1],J1[i1],H1[i1]])
      for i2 in range(2500):
        Jphot=J1[i1]-Al1[3]*Av[i2]
        Hphot=Jphot-JH_phot
        Rpphot=RpJ_phot+Jphot
        Gpphot=GpRp_phot+Rpphot
        Bpphot=BpRp_phot+Rpphot
        Magphot=np.array([Bpphot,Gpphot,Rpphot,Jphot,Hphot])      
        Diff[i2]=np.sqrt(np.sum((Magphot[sel1]-Mag1[sel1]+Al1[sel1]*Av[i2])**2))
      v1 = np.argmin(Diff)
      Avout[i1]=Av[v1]
      Dout[i1]=Diff[v1]

  if Mnorm == "J" and Jerr>errMlim:
    out1=np.full(5, np.nan)    

  if Mnorm == "Rp" and Rperr<=errMlim:
    All1=Al1[2]
    Mobs1=Rp1
    for i1 in range(nmc1):
      BC1[i1]= f_BCRp(T1[i1])
      JH_phot = f_JH(T1[i1])
      BpRp_phot = f_BpRp(T1[i1])
      GpRp_phot = f_GpRp(T1[i1])
      RpJ_phot = f_RpJ(T1[i1])
      Mag1=np.array([Bp1[i1],Gp1[i1],Rp1[i1],J1[i1],H1[i1]])
      for i2 in range(2500):
        Rpphot=Rp1[i1]-Al1[2]*Av[i2]
        Jphot=Rpphot-RpJ_phot
        Hphot=Jphot-JH_phot
        Gpphot=GpRp_phot+Rpphot
        Bpphot=BpRp_phot+Rpphot
        Magphot=np.array([Bpphot,Gpphot,Rpphot,Jphot,Hphot])      
        Diff[i2]=np.sqrt(np.sum((Magphot[sel1]-Mag1[sel1]+Al1[sel1]*Av[i2])**2))
      v1 = np.argmin(Diff)
      Avout[i1]=Av[v1]
      Dout[i1]=Diff[v1]
  if Mnorm == "Rp" and Rperr>errMlim:
    out1=np.full(5, np.nan)


  if Mnorm == "Gp" and Gperr<=errMlim:
    All1=Al1[1]
    Mobs1=Gp1
    for i1 in range(nmc1):
      BC1[i1]= f_BCGp(T1[i1])
      JH_phot = f_JH(T1[i1])
      BpRp_phot = f_BpRp(T1[i1])
      GpRp_phot = f_GpRp(T1[i1])
      RpJ_phot = f_RpJ(T1[i1])
      Mag1=np.array([Bp1[i1],Gp1[i1],Rp1[i1],J1[i1],H1[i1]])
      for i2 in range(2500):
        Gpphot=Gp1[i1]-Al1[1]*Av[i2]
        Rpphot=Gpphot-GpRp_phot
        Jphot=Rpphot-RpJ_phot
        Hphot=Jphot-JH_phot
        Bpphot=BpRp_phot+Rpphot
        Magphot=np.array([Bpphot,Gpphot,Rpphot,Jphot,Hphot])      
        Diff[i2]=np.sqrt(np.sum((Magphot[sel1]-Mag1[sel1]+Al1[sel1]*Av[i2])**2))
      v1 = np.argmin(Diff)
      Avout[i1]=Av[v1]
      Dout[i1]=Diff[v1]

  if Mnorm == "Gp" and Gperr>errMlim:
    out1=np.full(5, np.nan)  
  
  #CALCULO DE LA LUMINOSIDAD
  
  for i in range(nmc1):
  #    LgLout[i]=(4.7554-(J1[i]-5*math.log10(1000/PLX1[i])+5-Avout[i]*Al1[3])+BCJp)/2.5
    lgLout[i]=(4.7554- (Mobs1[i]-5.0*math.log10(1000/PLX1[i])+5.0-Avout[i]*All1+BC1[i]) )/2.5
  out1=np.array([Avout,Dout,T1,lgLout,npt1])
  #endT = time.time()
  #print(endT-staT, "seconds")
  #plt.figure(figsize=(8, 6))
  #sns.kdeplot(x=T1, y=Dout, shade=False, color="black", levels=5, thresh=0.1)
  #plt.scatter(T1,Dout)
  #plt.show()
  return out1
