import pandas as pd
import numpy as np
from massage_ext_f2 import massage_ext2
from massage_hrd_f3 import massage_hrd
from scipy.stats import skew

def massage_macroPLX (OBJECT,RA, DEC, Teff, eTeff, Bp, eBp, Gp, eGp, Rp, eRp, J, eJ, H, eH, PLX, ePLX, config_file):
  conf1 = pd.read_csv(config_file, sep=',')
#  print("RUNNING MASSAGE WITH:")
#  print(conf1)
  parameter = conf1['parameter'].to_numpy()
  value = conf1['option'].to_numpy()
  model1=value[0]
  phase1=value[1]
  inter_mode=value[2]
  min_mass=float(value[3])
  max_mass=float(value[4])
  min_age=float(value[5])
  max_age=float(value[6])
  nmc1=int(value[7])
  extlaw=value[8]
  Rv1=float(value[9])
  anch1=value[10]
  Xopt1=value[11]
  Yopt1=value[12]
  massagedir=value[13]
  output1=value[14]
  mc_output=value[15]
  #RUNNING EXTINCTION AND LUMINOSITY MODULUS
  tempo1=massage_ext2(Teff,eTeff,Xopt1,Bp,eBp,Gp,eGp,Rp,eRp,J,eJ,H,eH,anch1,Yopt1,nmc1,Rv1,extlaw,PLX,ePLX,inter_mode,massagedir)
  #RUNNING MASSAGE MODULUS
  tempo2=massage_hrd(tempo1[2,],tempo1[3,],model1,phase1,inter_mode,min_mass,max_mass,min_age,max_age,nmc1,massagedir)
  tempo2size=tempo2.size
  # SAVING INDIVIDUAL RESULTS
  if mc_output == "Y":
    tempfile1=output1 + "_" + str(OBJECT) + "_ext.csv"
    fileout1=''.join(tempfile1.split())
    np.savetxt(fileout1, np.transpose(tempo1), delimiter=',', header='Av,D_Av,Teff,logL,n_filters', comments='')
    if tempo2size>1:
      tempfile2=output1 + "_" + str(OBJECT) + "_massage.csv"
      fileout2=''.join(tempfile2.split())
      np.savetxt(fileout2, np.transpose(tempo2), delimiter=',', header='mass,age,logg', comments='')
  #PREPARING THE OUTPUT RESULTS
  #-------------Av---------------------
  Av_wmean =np.average(tempo1[0,],weights=1/tempo1[1,])
  eAv_wmean=np.sqrt(np.average((tempo1[0,] - Av_wmean)**2, weights=1/tempo1[1,]))
  Av_mean = np.mean(tempo1[0,])
  Av_std = np.std(tempo1[0,])
  Av_median = np.median(tempo1[0,])
  Av_mad = np.median( np.abs(tempo1[0,]-Av_median))
  Av_low = np.percentile(tempo1[0,], 15.8)
  Av_upp = np.percentile(tempo1[0,], 84.2)
  Av_skew = skew(tempo1[0,])
  #---------------Teff----------------
  T_mean = np.mean(tempo1[2,])
  T_std = np.std(tempo1[2,])
  T_median = np.median(tempo1[2,])
  T_mad = np.median( np.abs(tempo1[2,]-T_median))
  T_low = np.percentile(tempo1[2,], 15.8)
  T_upp = np.percentile(tempo1[2,], 84.2)
  T_skew = skew(tempo1[2,])
  #---------------lgL----------------
  lgL_mean = np.mean(tempo1[3,])
  lgL_std = np.std(tempo1[3,])
  lgL_median = np.median(tempo1[3,])
  lgL_mad = np.median( np.abs(tempo1[3,]-lgL_median))
  lgL_low = np.percentile(tempo1[3,], 15.8)
  lgL_upp = np.percentile(tempo1[3,], 84.2)
  lgL_skew = skew(tempo1[3,])
  #----------- mass ------------------

  if tempo2size>0:
    mass_mean = np.mean(tempo2[0,])
    mass_std = np.std(tempo2[0,])
    mass_median = np.median(tempo2[0,])
    mass_mad = np.median( np.abs(tempo2[0,]-mass_median))
    mass_low = np.percentile(tempo2[0,], 15.8)
    mass_upp = np.percentile(tempo2[0,], 84.2)
    mass_skew = skew(tempo2[0,])
  #----------- age ------------------
    age_mean = np.mean(tempo2[1,])
    age_std = np.std(tempo2[1,])
    age_median = np.median(tempo2[1,])
    age_mad = np.median( np.abs(tempo2[1,]-age_median))
    age_low = np.percentile(tempo2[1,], 15.8)
    age_upp = np.percentile(tempo2[1,], 84.2)
    age_skew = skew(tempo2[1,])
  #----------- lgg ------------------
    lgg_mean = np.mean(tempo2[2,])
    lgg_std = np.std(tempo2[2,])
    lgg_median = np.median(tempo2[2,])
    lgg_mad = np.median( np.abs(tempo2[2,]-lgg_median))
    lgg_low = np.percentile(tempo2[2,], 15.8)
    lgg_upp = np.percentile(tempo2[2,], 84.2)
    lgg_skew = skew(tempo2[2,])
    qflag=tempo2[3,0]
  if tempo2size==0:
    mass_mean = np.nan
    mass_std = np.nan
    mass_median = np.nan
    mass_mad = np.nan
    mass_low = np.nan
    mass_upp = np.nan
    mass_skew = np.nan
  #----------- age ------------------
    age_mean = np.nan
    age_std = np.nan
    age_median = np.nan
    age_mad = np.nan
    age_low = np.nan
    age_upp = np.nan
    age_skew = np.nan
  #----------- lgg ------------------
    lgg_mean = np.nan
    lgg_std = np.nan
    lgg_median = np.nan
    lgg_mad = np.nan
    lgg_low = np.nan
    lgg_upp = np.nan
    lgg_skew = np.nan
    qflag = -9
  # ----------------------
  perc_num=100*tempo2[0,].size/nmc1
  nfilter_1=tempo1[4,0]

  out3=np.array([OBJECT,RA,DEC,Av_wmean,eAv_wmean,Av_mean,Av_std, Av_median,1.4826*Av_mad,Av_low,Av_upp,Av_skew,nfilter_1,T_mean,T_std, T_median,1.4826*T_mad,T_low,T_upp,T_skew,lgL_mean,lgL_std, lgL_median,1.4826*lgL_mad,lgL_low,lgL_upp,lgL_skew, mass_mean,mass_std,mass_median,1.4826*mass_mad,mass_low,mass_upp,mass_skew,age_mean,age_std, age_median,1.4826*age_mad,age_low,age_upp,age_skew, lgg_mean,lgg_std, lgg_median,1.4826*lgg_mad,lgg_low,lgg_upp,lgg_skew,nmc1,perc_num,qflag])
  return out3
