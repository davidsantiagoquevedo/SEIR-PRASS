#!/usr/bin/python
# -*- coding:utf8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import odeint
BBDD_Dir='/home/santiago/Documentos/Desarrollo/Covid19/BBDD/'
"""
En este script se implementa la clase SEIR_PRASS referente al modelo de
Rastreo Aislamiento Selectivo Sostenible
"""

########################################################################
############### ########## SEIR_PRASS CLASS ########## #################
########################################################################
class SEIR_PRASS():
	def __init__(self):
		self.dummy=True
	
	def var_trans(self,beta_0,beta_1,beta_H,r,A,Movilidad,A2=False):
		self.beta_0=beta_0
		self.beta_1=beta_1
		self.beta_H=beta_H
		self.r=r
		self.A=A
		if  A2==False:
			self.A2=A
		else:
			self.A2=A2
		self.Movilidad=Movilidad
		
	def beta(self,t):
		if t<92:
			if self.A==0:
				At=self.A
				return (1-At)*self.beta_0 + At*self.beta_1		
			else:
				At=self.Movilidad[int(t)]
				return (1-At)*self.beta_0 + At*self.beta_1	
		if 92<=t<183:
			At=self.A
			return (1-At)*self.beta_0 + At*self.beta_1		
		if 183<=t:
			At=self.A2
			return (1-At)*self.beta_0 + At*self.beta_1		
	
	def var_t_estadia(self, omega, gamma_M, sigma_C, sigma_CA, gamma_HR, 
		nu, gamma_R, sigma_HD, sigma_UD):
		self.omega=omega		#T. prom. latencia
		self.gamma_M=gamma_M	#T. prom de recuperación para IM
		self.sigma_C=sigma_C	#T. antes de aislamiento IC
		self.sigma_CA=sigma_CA	#T. en aislamiento ICA
		self.gamma_HR=gamma_HR	#T. prom HR->R
		self.nu=nu				#T. prom HU->IR
		self.gamma_R=gamma_R	#T. prom IR->R
		self.sigma_HD=sigma_HD	#T. prom HD->D
		self.sigma_UD=sigma_UD	#T. prom UD->D
	
	def var_H_UCI(self, delta_M, delta_HR, delta_HD, delta_UR):
		self.delta_M=delta_M
		self.delta_HR=delta_HR
		self.delta_HD=delta_HD
		self.delta_UR=delta_UR
		self.delta_UD=1-delta_HR-delta_HD-delta_UR
		
	def var_testeo(self, xi_PCR=0, xi_RT=0, T_PCR=1., N_PCR=0, positividad=1):
		self.xi_PCR=xi_PCR
		self.xi_RT=xi_RT
		self.T_PCR=T_PCR
		self.N_PCR=N_PCR
		self.positividad=positividad
	
	def var_PRASS(self, alpha, rho, theta, q, theta_cons=True): 
		self.alpha=alpha
		self.theta=theta
		self.theta_cons=theta_cons
		self.rho=rho
		self.q=q
	
	def theta_t(self,t):
		if self.theta_cons==True:
			return self.theta
		else:
			if t<184:
				return 0.
			else:
				return self.theta		
		
	def var_ini(self, N0, E0=0, IM0=0, IMP0=0, IMNP0=0, IC0=0, ICA0=0, 
				IHR0=0, IUR0=0, IHD0=0, IUD0=0, IR0=0, R0=0, D0=0,
				F0=0, FT10=0, FTO0=0):
		self.N0=N0
		self.E0=E0
		
		self.IM0=IM0
		self.IMP0=IMP0
		self.IMNP0=IMNP0
		
		self.IC0=IC0
		self.ICA0=ICA0
				
		self.IHR0=IHR0
		self.IUR0=IUR0
		self.IHD0=IHD0
		self.IUD0=IUD0
		self.IR0=IR0
		self.R0=R0
		self.D0=D0
		
		self.F0=F0
		self.FT10=FT10
		self.FTO0=FTO0
		
		self.S0= (self.N0 - self.E0 - self.IM0 - self.IMP0  - self.IMNP0 
				  - self.IC0  - self.ICA0 - self.IHR0 - self.IUR0 - self.IHD0 
				  - self.IUD0 - self.IR0 - self.R0 - self.D0 - self.F0
				  - self.FT10 - self.FTO0)
	
	def ODES(self,y,t):
		S, E, IM, IMP, IMNP, IC, ICA, IHR, IUR, IHD, IUD, IR, R, D, N, F, FT1, FTO = y
		
		dSdt = -S/float(N)*(self.beta(t)*(IM+IC+(1-self.r)*ICA+IMNP+(1-self.q)*IMP)+self.beta_H*(IHR+IUR+IHD+IUD+IR))
		dEdt =  S/float(N)*(self.beta(t)*(IM+IC+(1-self.r)*ICA+IMNP+(1-self.q)*IMP)+self.beta_H*(IHR+IUR+IHD+IUD+IR)) - self.omega*E
		
		dIMdt = self.delta_M*self.omega*E - self.alpha*IM
		dIMPdt = self.theta_t(t)*self.alpha*IM - self.gamma_M*IMP
		dIMNPdt = (1-self.theta_t(t))*self.alpha*IM - self.gamma_M*IMNP
		
		dICdt = (1-self.delta_M)*self.omega*E - self.sigma_C*IC
		dICAdt = self.sigma_C*IC - self.sigma_CA*ICA
				
		dIHRdt = self.delta_HR*self.sigma_CA*ICA - self.gamma_HR*IHR
		dIURdt = self.delta_UR*self.sigma_CA*ICA - self.nu*IUR
		dIHDdt = self.delta_HD*self.sigma_CA*ICA - self.sigma_HD*IHD
		dIUDdt = self.delta_UD*self.sigma_CA*ICA - self.sigma_UD*IUD
		
		dIRdt = self.nu*IUR - self.gamma_R*IR
		dRdt = self.gamma_HR*IHR + self.gamma_R*IR + self.gamma_M*IMP + self.gamma_M*IMNP
		
		dDdt = self.sigma_HD*IHD + self.sigma_UD*IUD
		dNdt = -self.sigma_HD*IHD - self.sigma_UD*IUD
		
		dFdt = self.theta_t(t)*self.alpha*IM - 1/float(self.T_PCR)*F
		dFT1dt = self.xi_PCR*1/float(self.T_PCR)*F - self.rho*FT1
		dFTOdt = (1-self.xi_PCR)*1/float(self.T_PCR)*F 
		
		return [dSdt, dEdt, dIMdt, dIMPdt, dIMNPdt, dICdt, dICAdt, 
				dIHRdt, dIURdt, dIHDdt, dIUDdt, dIRdt, dRdt, dDdt, dNdt,
				dFdt, dFT1dt , dFTOdt]
		
	def solve(self,t0,tf,dt):
		self.t0=t0
		self.tf=tf
		self.dt_=1/dt
		y0= [self.S0, self.E0, self.IM0, self.IMP0, self.IMNP0, 
			 self.IC0, self.ICA0, self.IHR0, self.IUR0, 
			 self.IHD0, self.IUD0, self.IR0, self.R0, self.D0, self.N0,
			 self.F0, self.FT10, self.FTO0]
		t_vect= np.linspace(self.t0, self.tf, self.tf*self.dt_)
		self.t_=t_vect
		solution= odeint(self.ODES,y0,t_vect)
		
		self.S_vect=solution.T[0]
		self.E_vect=solution.T[1]
		
		self.IM_vect=solution.T[2]
		self.IMP_vect=solution.T[3]
		self.IMNP_vect=solution.T[4]
		
		self.IC_vect=solution.T[5]
		self.ICA_vect=solution.T[6]
				
		self.IHR_vect=solution.T[7]
		self.IUR_vect=solution.T[8]
		self.IHD_vect=solution.T[9]
		self.IUD_vect=solution.T[10]
		
		self.IR_vect=solution.T[11]
		self.R_vect=solution.T[12]
		self.D_vect=solution.T[13]
		self.N_vect=solution.T[14]
		
		self.F_vect=solution.T[15]
		self.FT1_vect=solution.T[16]
		self.FTO_vect=solution.T[17]
		
	def Contar_Tests(self):
		self.PCR_IC=[]
		self.PCR_IM=[]
		self.RT=[]
		for i in range(len(self.t_)):
			if self.t_[i]<122: pos=self.positividad[0] #junio
			if 122<=self.t_[i]<153: pos=self.positividad[1] #julio
			if 153<=self.t_[i]<184: pos=self.positividad[2] #agosto
			if 184<=self.t_[i]: pos=self.positividad[3] #septiembre
			if self.t_[i]<153:
				self.PCR_IC.append(self.sigma_CA*self.ICA_vect[i]*pos)
				self.PCR_IM.append(self.alpha*self.theta_t(self.t_[i])*self.IM_vect[i]*pos)
				self.RT.append(0)
			if self.t_[i]>=153:
				self.PCR_IC.append((1-self.xi_RT)*self.sigma_CA*self.ICA_vect[i]*pos)
				self.PCR_IM.append(self.alpha*self.theta_t(self.t_[i])*self.IM_vect[i]*pos)
				self.RT.append(self.sigma_CA*self.ICA_vect[i]*pos)
		self.PCR_Total=np.array(self.PCR_IC)+np.array(self.PCR_IM)
				
	def Contar_PRASS(self):
		self.PRASS_incidencia=[]
		self.PRASS_prevalencia=[]
		for i in range(len(self.t_)):
			if self.t_[i]<122: pos=self.positividad[0] #junio
			if 122<=self.t_[i]<153: pos=self.positividad[1] #julio
			if 153<=self.t_[i]<183: pos=self.positividad[2] #agosto
			if 183<=self.t_[i]: pos=self.positividad[3] #septiembre
			self.PRASS_incidencia.append(self.alpha*self.theta_t(self.t_[i])*self.IM_vect[i])
			self.PRASS_prevalencia.append(self.F_vect[i]+self.FT1_vect[i])
			
	def Contar_Rastreo(self):
		self.t_rastreo_in=45.
		self.t_rastreo_prev=20.
		self.promedio_contactos=2.
		self.unidad_rastreo=360.
		
		self.cuenta_in=[]
		self.cuenta_prev=[]
		self.Contar_PRASS()
		
		for i in range(len(self.t_)):
			self.cuenta_in.append(self.PRASS_incidencia[i]*self.t_rastreo_in)
			self.cuenta_prev.append(self.PRASS_prevalencia[i]*self.t_rastreo_prev*self.promedio_contactos
							   -self.PRASS_prevalencia[i]*self.t_rastreo_prev)
		self.Total_Rastreo=(np.array(self.cuenta_in)+np.array(self.cuenta_prev))/self.unidad_rastreo
	
	def Contar_UCI(self):
		self.Requieren_UCI=np.array(self.IUR_vect)+np.array(self.IUD_vect)
		
	def plot_tests(self):
		fig = plt.figure(figsize=(18, 16), dpi= 30, facecolor='w', edgecolor='k')
		ax = plt.subplot(111)
		self.Contar_Tests()
		nombre1=r'Testeados PCR - $I_M$'; Data=np.array(self.PCR_IM)
		nombre2=r'Testeados PCR - $I_C$'; Data2=np.array(self.PCR_IC)
		nombre3='Testeados PCR - Total'; Data3=(Data+Data2)
		nombre4='Testeados RT'; Data4=np.array(self.RT)
		nombre='Testeados'
		t_vect=np.linspace(self.t0,self.tf,len(Data))
		plt.title(nombre,fontsize=34)
		plt.ylabel(u"# Tests", weight='bold', fontsize=24)
		ax.plot(t_vect,Data4, linestyle='-', marker=' ', label=str(nombre4), linewidth=4.0, color='firebrick')
		ax.plot(t_vect,Data3, linestyle='-', marker=' ', label=str(nombre3), linewidth=4.0, color='green')
		ax.plot(t_vect,Data, linestyle='--', marker=' ', label=str(nombre1), linewidth=4.0, color='royalblue')
		ax.plot(t_vect,Data2, linestyle='--', marker=' ', label=str(nombre2), linewidth=4.0, color='orange')
		plt.xlabel(u"Días", weight='bold', fontsize=24)
		#ax.ticklabel_format(axis='y',style='sci',scilimits=(0,3),fontsize=24)
		plt.rc('font', **{'size':'30'})
		plt.yticks(size=24)
		plt.xticks(size=24)
		ax.legend(loc='best',fontsize=24)
		
		loc=-max([max(Data),max(Data2),max(Data3),max(Data4)])/10.
		ax.get_xaxis().set_visible(False)		
		plt.axvline(x=61, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(70,loc, 'May-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=92, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(101,loc, 'Jun-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=122, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(131,loc, 'Jul-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=153, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(162,loc, 'Ago-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=184, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(193,loc, 'Sep-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=214, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(223,loc, 'Oct-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=245, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(254,loc, 'Nov-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=275, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(284,loc, 'Dic-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=306, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(315,loc, 'Ene-21', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=337, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(346,loc, 'Feb-21', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=365, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(374,loc, 'Mar-21', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=396, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		plt.show()
		
	def plot_UCI(self):
		fig = plt.figure(figsize=(18, 16), dpi= 30, facecolor='w', edgecolor='k')
		ax = plt.subplot(111)
		self.Contar_UCI()
		nombre='Requieren UCI'; Data=self.Requieren_UCI
		t_vect=np.linspace(self.t0,self.tf,len(Data))
		plt.title(nombre,fontsize=34)
		plt.ylabel(u"Cantidad de UCIS", weight='bold', fontsize=24)
		ax.plot(t_vect,Data, linestyle='--', marker=' ', linewidth=4.0, color='royalblue')
		plt.xlabel(u"Días", weight='bold', fontsize=24)
		#ax.ticklabel_format(axis='y',style='sci',scilimits=(0,3),fontsize=24)
		plt.rc('font', **{'size':'30'})
		plt.yticks(size=24)
		plt.xticks(size=24)
		ax.legend(loc='best',fontsize=24)
		
		loc=-max(Data)/10.
		ax.get_xaxis().set_visible(False)		
		plt.axvline(x=61, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(70,loc, 'May-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=92, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(101,loc, 'Jun-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=122, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(131,loc, 'Jul-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=153, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(162,loc, 'Ago-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=184, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(193,loc, 'Sep-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=214, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(223,loc, 'Oct-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=245, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(254,loc, 'Nov-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=275, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(284,loc, 'Dic-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=306, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(315,loc, 'Ene-21', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=337, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(346,loc, 'Feb-21', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=365, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(374,loc, 'Mar-21', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=396, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		
		plt.show()
			
	def plot_PRASS(self):
		fig = plt.figure(figsize=(18, 16), dpi= 30, facecolor='w', edgecolor='k')
		ax = plt.subplot(111)
		self.Contar_Rastreo()
		nombre='PRASSEADOS'; 
		Data=np.array(self.PRASS_incidencia)
		Data1=np.array(self.PRASS_prevalencia)
		t_vect=np.linspace(self.t0,self.tf,len(Data))
		plt.title(nombre,fontsize=34)
		plt.ylabel(u'# Infecciosos seguidos', fontsize=24)
		ax.plot(t_vect,Data, linestyle='-', marker=' ', linewidth=4.0, color='royalblue',label='Incidencia')
		ax.plot(t_vect,Data1, linestyle='--', marker=' ', linewidth=4.0, color='red',label='Prevalencia')
		plt.xlabel(u"Días", weight='bold', fontsize=24)
		#ax.ticklabel_format(axis='y',style='sci',scilimits=(0,3),fontsize=24)
		plt.rc('font', **{'size':'30'})
		plt.yticks(size=24)
		plt.xticks(size=24)
		ax.legend(loc='best',fontsize=24)
		
		loc=-max([max(Data),max(Data1)])/10.
		ax.get_xaxis().set_visible(False)		
		plt.axvline(x=61, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(70,loc, 'May-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=92, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(101,loc, 'Jun-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=122, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(131,loc, 'Jul-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=153, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(162,loc, 'Ago-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=184, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(193,loc, 'Sep-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=214, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(223,loc, 'Oct-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=245, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(254,loc, 'Nov-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=275, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(284,loc, 'Dic-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=306, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(315,loc, 'Ene-21', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=337, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(346,loc, 'Feb-21', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=365, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(374,loc, 'Mar-21', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=396, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		plt.show()
	
	def  plot_Rastreadores(self):
		fig = plt.figure(figsize=(18, 16), dpi= 30, facecolor='w', edgecolor='k')
		ax = plt.subplot(111)
		self.Contar_Rastreo()
		nombre='Rastreadores'; Data=np.array(self.Total_Rastreo)
		t_vect=np.linspace(self.t0,self.tf,len(Data))
		plt.title(nombre,fontsize=34)
		plt.ylabel(u"# Rastreadores", weight='bold', fontsize=24)
		ax.plot(t_vect,Data, linestyle='-', marker=' ', linewidth=4.0, color='royalblue')
		plt.xlabel(u"Días", weight='bold', fontsize=24)
		#ax.ticklabel_format(axis='y',style='sci',scilimits=(0,3),fontsize=24)
		plt.rc('font', **{'size':'30'})
		plt.yticks(size=24)
		plt.xticks(size=24)
		ax.legend(loc='best',fontsize=24)
		
		loc=-max(Data)/10.
		ax.get_xaxis().set_visible(False)		
		plt.axvline(x=61, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(70,loc, 'May-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=92, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(101,loc, 'Jun-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=122, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(131,loc, 'Jul-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=153, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(162,loc, 'Ago-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=184, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(193,loc, 'Sep-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=214, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(223,loc, 'Oct-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=245, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(254,loc, 'Nov-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=275, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(284,loc, 'Dic-20', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=306, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(315,loc, 'Ene-21', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=337, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(346,loc, 'Feb-21', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=365, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		ax.text(374,loc, 'Mar-21', weight='bold', fontsize=24, rotation=90)
		plt.axvline(x=396, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
		plt.show()

def plot_varios(x,y, eje_x, eje_y, labels,nombre=False):
	fig = plt.figure(figsize=(18, 16), dpi= 30, facecolor='w', edgecolor='k')
	ax = plt.subplot(111)
	if nombre:
		plt.title(nombre,fontsize=34)
	mx=[]
	for i in range(len(x)):
		mx.append(max(y[i]))
		ax.plot(x[i],y[i], linestyle='--', marker=' ', linewidth=4.0,label=labels[i])
	plt.xlabel(eje_x, weight='bold', fontsize=24)
	plt.ylabel(eje_y, weight='bold', fontsize=24)
	#ax.ticklabel_format(axis='y',style='sci',scilimits=(0,3),fontsize=24)
	plt.rc('font', **{'size':'30'})
	plt.yticks(size=24)
	plt.xticks(size=24)
	ax.legend(loc='best',fontsize=24)
	
	loc=-max(mx)/10.
	ax.get_xaxis().set_visible(False)		
	plt.axvline(x=61, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
	ax.text(70,loc, 'May-20', weight='bold', fontsize=24, rotation=90)
	plt.axvline(x=92, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
	ax.text(101,loc, 'Jun-20', weight='bold', fontsize=24, rotation=90)
	plt.axvline(x=122, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
	ax.text(131,loc, 'Jul-20', weight='bold', fontsize=24, rotation=90)
	plt.axvline(x=153, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
	ax.text(162,loc, 'Ago-20', weight='bold', fontsize=24, rotation=90)
	plt.axvline(x=184, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
	ax.text(193,loc, 'Sep-20', weight='bold', fontsize=24, rotation=90)
	plt.axvline(x=214, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
	ax.text(223,loc, 'Oct-20', weight='bold', fontsize=24, rotation=90)
	plt.axvline(x=245, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
	ax.text(254,loc, 'Nov-20', weight='bold', fontsize=24, rotation=90)
	plt.axvline(x=275, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
	ax.text(284,loc, 'Dic-20', weight='bold', fontsize=24, rotation=90)
	plt.axvline(x=306, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
	ax.text(315,loc, 'Ene-21', weight='bold', fontsize=24, rotation=90)
	plt.axvline(x=337, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
	ax.text(346,loc, 'Feb-21', weight='bold', fontsize=24, rotation=90)
	plt.axvline(x=365, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
	ax.text(374,loc, 'Mar-21', weight='bold', fontsize=24, rotation=90)
	plt.axvline(x=396, ymin=0, ymax=1, color='grey', linestyle='--', linewidth=1.0)
	plt.show()
			
			
		
