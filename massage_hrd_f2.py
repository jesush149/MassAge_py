# A function version of the modulus to obtain age and masses from a HRD point (Teff, logL)
import sys
import os
import math
import time
import random
import pandas as pd
import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import griddata

staT = time.time()

def massage_hrd (T1, lgL1, modelo, phase_sel,inter_mode,min_mass,max_mass,min_age,max_age,nmc1,massagedir):
#OPCIONES from configuration file.
#  conf1 = pd.read_csv(conf_file1, sep=',')
#  modelo=np.array(conf1[['modelo']])[0]
#  phase_sel=np.array(conf1[['phase_sel']])[0]
#  inter_mode=np.array(conf1[['inter_mode']])[0]
#  min_mass=np.array(conf1[['min_mass']])[0]
#  max_mass=np.array(conf1[['max_mass']])[0]
#  min_age=np.array(conf1[['min_age']])[0]
#  max_age=np.array(conf1[['max_age']])[0]
#  nmc1=np.array(conf1[['num_nmc1']])[0]

  #MODELO PARSEC includes different evolutionary states 
  if modelo == "PARSEC":
    moddir="MODELS/PARSEC/PARSEC_logage_001_V3.csv"
    file1 = os.path.join(massagedir, moddir)
    dat1 = pd.read_csv(file1, sep=',')
    #Read model values
    lgT_o = np.array(dat1[['logTe']])
    T_o = 10**lgT_o
    lgL_o=np.array(dat1[['logL']])
    lgage_o = np.array(dat1[['logAge']])
    age_o = 10**lgage_o/1.0e6
    mass_o = np.array(dat1[['Mass']])
    logg_o = np.array(dat1[['logg']])
    label_o = np.array(dat1[['label']])
    # 0) pre-main sequence, 1) main sequence, 2) subgiant branch, 3) RGB, 4) CHeB, 5) early-AGB, 6) TPAGB
    phase1=100
    if phase_sel=='pms' or phase_sel=='ms':
        phase = {
        'pms': 0,
        'ms': 1,
        'all': 100
        }
        phase1=phase[phase_sel]

    i1=np.where((label_o<=phase1) & (mass_o<=max_mass) & (mass_o>=min_mass) & (age_o<=max_age) & (age_o>=min_age))
    Teff=T_o[i1]
    lgL=lgL_o[i1]
    age=age_o[i1]
    mass=mass_o[i1]
    logg=logg_o[i1]

  #MODELO MIST includes different evolutionary states 
  if modelo == "MIST":
    moddir="MODELS/MIST/MIST_V2.csv"
    file1 = os.path.join(massagedir, moddir)
    dat1 = pd.read_csv(file1, sep=',')
    #Read model values
    lgT_o = np.array(dat1[['log_Teff']])
    T_o = 10**lgT_o
    lgL_o=np.array(dat1[['log_L']])
    age_o = np.array(dat1[['star_age']])/1.0e6
    mass_o = np.array(dat1[['star_mass']])
    logg_o = np.array(dat1[['log_g']])
    label_o = np.array(dat1[['phase']])
    #-1=PMS, 0=MS, 2=RGB, 3=CHeB, 4=EAGB, 5=TPAGB, 6=postAGB, 9=WR
    phase1=100
    if phase_sel=='pms' or phase_sel=='ms':
        phase = {
        'pms': -1,
        'ms': 0,
        'all': 100
        }
        phase1=phase[phase_sel]

    i1=np.where((label_o<=phase1) & (mass_o<=max_mass) & (mass_o>=min_mass) & (age_o<=max_age) & (age_o>=min_age))
    Teff=T_o[i1]
    lgL=lgL_o[i1]
    age=age_o[i1]
    mass=mass_o[i1]
    logg=logg_o[i1]

  #MODELO SF00 
  if modelo == "SF00":
    moddir="MODELS/SF00/SF00_v1.csv"
    file1 = os.path.join(massagedir, moddir)
    dat1 = pd.read_csv(file1, sep=',')
    #Read model values
    lgT_o = np.array(dat1[['logTe']])
    T_o = 10**lgT_o
    lgL_o=np.array(dat1[['logL']])
    lgage_o = np.array(dat1[['logAge']])
    age_o = 10**lgage_o/1.0e6
    mass_o = np.array(dat1[['mass2']])
    logg_o = np.array(dat1[['logg']])
    label_o = np.array(dat1[['phase']])
    # PMS=1; MS=2
    phase1=100
    if phase_sel=='pms' or phase_sel=='ms':
        phase = {
        'pms': 1,
        'ms': 2,
        'all': 100
        }
        phase1=phase[phase_sel]
    
    i1=np.where( (label_o<=phase1) & (mass_o<=max_mass) & (mass_o>=min_mass) & (age_o<=max_age) & (age_o>=min_age))
    Teff=T_o[i1]
    lgL=lgL_o[i1]
    age=age_o[i1]
    mass=mass_o[i1]
    logg=logg_o[i1]

  #MODELO PISA does not include different evolutionary states, logg estimated from Lum,Teff & Mass 
  if modelo == "PISA":
    moddir="MODELS/PISA/PISA_v1.csv"
    file1 = os.path.join(massagedir, moddir)
    dat1 = pd.read_csv(file1, sep=',')
    #Read model values
    lgT_o = np.array(dat1[['lgTeff']])
    T_o = 10**lgT_o
    lgL_o=np.array(dat1[['lgLum']])
    age_o = np.array(dat1[['age']])/1.0e6
    mass_o = np.array(dat1[['Mass']])
    
    ####estimating logg in CGS
    Rsun=6.96e10
    Lsun=3.847e33
    Msun=1.9891e33
    sb_sig=5.67051e-5
    Gc=6.6726e-8
    logg_o=np.log10(Gc)+np.log10(mass_o*Msun)-lgL_o-np.log10(Lsun)+np.log10(4*math.pi*sb_sig)+4*np.log10(T_o)
    
    i1=np.where((mass_o<=max_mass) & (mass_o>=min_mass) & (age_o<=max_age) & (age_o>=min_age))
    Teff=T_o[i1]
    lgL=lgL_o[i1]
    age=age_o[i1]
    mass=mass_o[i1]
    logg=logg_o[i1]

  #MODELO Baraffe+2015 does not include different evolutionary states 
  if modelo == "Baraffe":
    moddir="MODELS/BARAFFE/BARAFFE.csv"
    file1 = os.path.join(massagedir, moddir)
    dat1 = pd.read_csv(file1, sep=',')
    #Read model values
    lgT_o = np.array(dat1[['logTeff']])
    T_o = 10**lgT_o
    lgL_o=np.array(dat1[['LogLum']])
    age_o = np.array(dat1[['age']])
    mass_o = np.array(dat1[['mass']])
    logg_o = np.array(dat1[['g']])
    i1=np.where( (mass_o<=max_mass) & (mass_o>=min_mass) & (age_o<=max_age) & (age_o>=min_age))
    Teff=T_o[i1]
    lgL=lgL_o[i1]
    age=age_o[i1]
    mass=mass_o[i1]
    logg=logg_o[i1]

  #MODELO Feiden standard does not include different evolutionary states 
  if modelo == "Feiden_std":
    moddir="MODELS/FEIDEN_std/Feiden_Std.csv"
    file1 = os.path.join(massagedir, moddir)
    dat1 = pd.read_csv(file1, sep=',')
    #Read model values
    lgT_o = np.array(dat1[['lgTemp']])
    T_o = 10**lgT_o
    lgL_o=np.array(dat1[['lgLum']])
    age_o = np.array(dat1[['age']])/1.0e6
    mass_o = np.array(dat1[['mass']])
    logg_o = np.array(dat1[['logg']])
    i1=np.where( (mass_o<=max_mass) & (mass_o>=min_mass) & (age_o<=max_age) & (age_o>=min_age))
    Teff=T_o[i1]
    lgL=lgL_o[i1]
    age=age_o[i1]
    mass=mass_o[i1]
    logg=logg_o[i1]                      
  #MODELO Feiden magnetic does not include different evolutionary states
  if modelo == "Feiden_mag":
    moddir="MODELS/FEIDEN_mag/Feiden_Mag.csv"
    file1 = os.path.join(massagedir, moddir)
    dat1 = pd.read_csv(file1, sep=',')
    #Read model values
    lgT_o = np.array(dat1[['lgTemp']])
    T_o = 10**lgT_o
    lgL_o=np.array(dat1[['lgLum']])
    age_o = np.array(dat1[['age']])/1.0e6
    mass_o = np.array(dat1[['mass']])
    logg_o = np.array(dat1[['logg']])
    i1=np.where( (mass_o<=max_mass) & (mass_o>=min_mass) & (age_o<=max_age) & (age_o>=min_age))
    Teff=T_o[i1]
    lgL=lgL_o[i1]
    age=age_o[i1]
    mass=mass_o[i1]
    logg=logg_o[i1]        
  #MODELO Somers+2020  no spot
  if modelo == "Somers_0":
    moddir="MODELS/Somers/f000track_ms.csv"
    file1 = os.path.join(massagedir, moddir)
    dat1 = pd.read_csv(file1, sep=',')
    #Read model values
    lgT_o = np.array(dat1[['log(Teff)']])
    T_o = 10**lgT_o
    lgL_o=np.array(dat1[['log(L/Lsun)']])
    age_o = np.array(dat1[['Age(Gyr)']])*1.0e3
    mass_o = np.array(dat1[['Mass']])
    logg_o = np.array(dat1[['log(g)']])
    i1=np.where( (mass_o<=max_mass) & (mass_o>=min_mass) & (age_o<=max_age) & (age_o>=min_age))
    Teff=T_o[i1]
    lgL=lgL_o[i1]
    age=age_o[i1]
    mass=mass_o[i1]
    logg=logg_o[i1]     
  #MODELO Somers+2020  17% spot
  if modelo == "Somers_17":
    moddir="MODELS/Somers/f017track_ms.csv"
    file1 = os.path.join(massagedir, moddir)
    dat1 = pd.read_csv(file1, sep=',')
    #Read model values
    lgT_o = np.array(dat1[['log(Teff)']])
    T_o = 10**lgT_o
    lgL_o=np.array(dat1[['log(L/Lsun)']])
    age_o = np.array(dat1[['Age(Gyr)']])*1.0e3
    mass_o = np.array(dat1[['Mass']])
    logg_o = np.array(dat1[['log(g)']])
    i1=np.where( (mass_o<=max_mass) & (mass_o>=min_mass) & (age_o<=max_age) & (age_o>=min_age))
    Teff=T_o[i1]
    lgL=lgL_o[i1]
    age=age_o[i1]
    mass=mass_o[i1]
    logg=logg_o[i1]
     
  #MODELO Somers+2020  34% spot    
  if modelo == "Somers_34":
    moddir="MODELS/Somers/f034track_ms.csv"
    file1 = os.path.join(massagedir, moddir)
    dat1 = pd.read_csv(file1, sep=',')
    #Read model values
    lgT_o = np.array(dat1[['log(Teff)']])
    T_o = 10**lgT_o
    lgL_o=np.array(dat1[['log(L/Lsun)']])
    age_o = np.array(dat1[['Age(Gyr)']])*1.0e3
    mass_o = np.array(dat1[['Mass']])
    logg_o = np.array(dat1[['log(g)']])
    i1=np.where( (mass_o<=max_mass) & (mass_o>=min_mass) & (age_o<=max_age) & (age_o>=min_age))
    Teff=T_o[i1]
    lgL=lgL_o[i1]
    age=age_o[i1]
    mass=mass_o[i1]
    logg=logg_o[i1]     

  #MODELO Somers+2020  51% spot    
  if modelo == "Somers_51":
    moddir="MODELS/Somers/f051track_ms.csv"
    file1 = os.path.join(massagedir, moddir)
    dat1 = pd.read_csv(file1, sep=',')
    #Read model values
    lgT_o = np.array(dat1[['log(Teff)']])
    T_o = 10**lgT_o
    lgL_o=np.array(dat1[['log(L/Lsun)']])
    age_o = np.array(dat1[['Age(Gyr)']])*1.0e3
    mass_o = np.array(dat1[['Mass']])
    logg_o = np.array(dat1[['log(g)']])
    i1=np.where( (mass_o<=max_mass) & (mass_o>=min_mass) & (age_o<=max_age) & (age_o>=min_age))
    Teff=T_o[i1]
    lgL=lgL_o[i1]
    age=age_o[i1]
    mass=mass_o[i1]
    logg=logg_o[i1]     

  #MODELO Somers+2020  68% spot
  if modelo == "Somers_68":
    moddir="MODELS/Somers/f068track_ms.csv"
    file1 = os.path.join(massagedir, moddir)
    dat1 = pd.read_csv(file1, sep=',')
    #Read model values
    lgT_o = np.array(dat1[['log(Teff)']])
    T_o = 10**lgT_o
    lgL_o=np.array(dat1[['log(L/Lsun)']])
    age_o = np.array(dat1[['Age(Gyr)']])*1.0e3
    mass_o = np.array(dat1[['Mass']])
    logg_o = np.array(dat1[['log(g)']])
    i1=np.where( (mass_o<=max_mass) & (mass_o>=min_mass) & (age_o<=max_age) & (age_o>=min_age))
    Teff=T_o[i1]
    lgL=lgL_o[i1]
    age=age_o[i1]
    mass=mass_o[i1]
    logg=logg_o[i1]     
    
  #MODELO Somers+2020  85% spot
  if modelo == "Somers_85":
    moddir="MODELS/Somers/f085track_ms.csv"
    file1 = os.path.join(massagedir, moddir)
    dat1 = pd.read_csv(file1, sep=',')
    #Read model values
    lgT_o = np.array(dat1[['log(Teff)']])
    T_o = 10**lgT_o
    lgL_o=np.array(dat1[['log(L/Lsun)']])
    age_o = np.array(dat1[['Age(Gyr)']])*1.0e3
    mass_o = np.array(dat1[['Mass']])
    logg_o = np.array(dat1[['log(g)']])
    i1=np.where( (mass_o<=max_mass) & (mass_o>=min_mass) & (age_o<=max_age) & (age_o>=min_age))
    Teff=T_o[i1]
    lgL=lgL_o[i1]
    age=age_o[i1]
    mass=mass_o[i1]
    logg=logg_o[i1]     
    

  # Si restringimos la malla entre los valores a interpolar se disminuira el tiempo de computo.
  #checking if the theoretical grid has at least 5 points within 3sigma from the observed point
  ### aquí hay la posibilidad de encontrar vacios en la maya dentro de los límites de  la misma, 
  ## principalmente cuando los errores son muy pequeños
  flag_q=0
  nsig=10
  Teff_s=np.mean(T1)
  eTeff_s=np.std(T1)
  lgL_s=np.mean(lgL1)
  elgL_s=np.std(lgL1)
  print(Teff_s,eTeff_s,lgL_s,elgL_s)
  i0=np.where((Teff<=Teff_s+nsig*eTeff_s) & (Teff>=Teff_s-nsig*eTeff_s) & (lgL<=lgL_s+nsig*elgL_s) & (lgL>=lgL_s-nsig*elgL_s))
  num_i0=np.size(i0)
  print(num_i0)
  if num_i0>5:
    flag_q=1

  #theoretical Grid
  points = np.column_stack((Teff[i0], lgL[i0]))
  values_mass = mass[i0]
  values_age = age[i0]
  values_logg = logg[i0]
  values_Teff = Teff[i0]
  values_lgL  = lgL[i0]

  #Points to be interpolated 
  temp2=T1
  lglum2=lgL1
  grid_points = np.column_stack((temp2.ravel(), lglum2.ravel()))


  #interpoling, it takes some times, almost independent of points to be interpolated
  out_mass = griddata(points, values_mass, grid_points, method=inter_mode)
  out_ages = griddata(points, values_age, grid_points, method=inter_mode)
  out_logg = griddata(points, values_logg, grid_points, method=inter_mode)

  #rejecting values outside from the imposed limits 
  i2=np.where((out_mass<=max_mass) & (out_mass>=min_mass) & (out_ages<=max_age) & (out_ages>=min_age))
  num_i2=np.size(i2)

  outmass=out_mass[i2]
  outages=out_ages[i2]
  outlogg=out_logg[i2]
  outtemp=temp2[i2]
  outlglum=lglum2[i2]

  # it requires at least 5 MC points within the limits in mass and age
  if num_i2 >5:
    lol_mass = np.percentile(outmass, 15.9)  # -1 sig gaussian equivalent
    mid_mass = np.percentile(outmass, 50)    # median
    upl_mass = np.percentile(outmass, 84.1)  # +1 sig gaussian equivalent
    skew_mass = skew(outmass, bias=False)       # skewness, bias must be False
    lol_ages = np.percentile(outages, 15.9)  # -1 sig gaussian equivalent
    mid_ages = np.percentile(outages, 50)    # median
    upl_ages = np.percentile(outages, 84.1)  # +1 sig gaussian equivalent
    skew_ages = skew(outages, bias=False)       # skewness, bias must be False
    lol_logg = np.percentile(outlogg, 15.9)  # -1 sig gaussian equivalent
    mid_logg = np.percentile(outlogg, 50)    # median
    upl_logg = np.percentile(outlogg, 84.1)  # +1 sig gaussian equivalent
    skew_logg = skew(outlogg, bias=False)       # skewness, bias must be False
    meanTeff = np.mean(outtemp)
    meanLgL  = np.mean(outlglum)
    stdTeff = np.std(outtemp)
    stdLgL  = np.std(outlglum)
    frac_num = (num_i2/nmc1)
    flag_q = 1
  if num_i2 <=5:
    lol_mass = np.nan  # -1 sig gaussian equivalent
    mid_mass = np.nan    # median
    upl_mass = np.nan  # +1 sig gaussian equivalent
    skew_mass = np.nan       # skewness, bias must be False
    lol_ages = np.nan  # -1 sig gaussian equivalent
    mid_ages = np.nan    # median
    upl_ages = np.nan  # +1 sig gaussian equivalent
    skew_ages = np.nan       # skewness, bias must be False
    lol_logg = np.nan  # -1 sig gaussian equivalent
    mid_logg = np.nan    # median
    upl_logg = np.nan  # +1 sig gaussian equivalent
    skew_logg = np.nan       # skewness, bias must be False
    meanTeff = Teff_s
    meanLgL  = lgL_s
    stdTeff = eTeff_s
    stdLgL  = elgL_s
    frac_num = (num_i2/nmc1)
    flag_q = 0
  out1=[meanTeff,stdTeff,meanLgL,stdLgL,mid_mass,lol_mass,upl_mass,skew_mass,mid_ages,lol_ages,upl_ages,skew_ages, mid_logg, lol_logg, upl_logg,skew_logg, frac_num,flag_q]
  endT = time.time()
  print(endT-staT, "seconds")
  return out1
