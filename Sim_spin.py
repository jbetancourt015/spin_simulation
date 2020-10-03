# -*- coding: utf-8 -*-
"""
Simulación de Monte Carlo para el modelo de Ising en 2D

Creado por Jose Betancourt el 27 de septiembre del 2020
"""
#Importación de paquetes
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.special import ellipk
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Garamond"],
})

#Definición del directorio
os.chdir("C:\\Users\\USER\\OneDrive - Universidad de los Andes\\Undergrad\\2020-2\\Mecánica Estadística")

#Parámetros del modelo
n = 50 #Número de espines en cada lado de la red
S = 20000000 #Número de iteraciones para cada instancia de la temperatura
T_m = 4 #Valor máximo de temperatura que se analizará
n_T = 121 #Longitud de los pasos iterados sorbe beta

#Rengo de temperaturas utilizado
T_vals = np.linspace(0,T_m,n_T)[1:]

#Función para calcular la energía inicial de una red
def E_in(sp,L):
    ny = sp.shape[0]
    nx = sp.shape[1]
    E0 = 0
    for i in range(nx):
        for j in range(ny):
            E0 += (L/2)*(sp[i,j]*sp[i,(j+1)%nx] + sp[i,j]*sp[i,(j-1)%nx] + sp[i,j]*sp[(i+1)%ny,j] + sp[i,j]*sp[(i-1)%ny,j])
    return E0


#Iteración sobre diferentes valores de la temperatura
counter = 0

#Arreglos para guardar los promedios
E_prom = []
E_std = []
M_prom = []
M_std = []

for T in T_vals:
    #Inicialización de la red
    red = np.random.choice([-1,1],[n,n])
    
    L = 1/T
    E = E_in(red,L)
    M = np.sum(red)
    En = []
    Mn = []

    #Evolución utilizando el algoritmo
    for t in range(S):
        i = np.random.randint(n)
        j = np.random.randint(n)
        DE = 2*L*(red[i,j]*red[i,(j+1)%n] + red[i,j]*red[i,(j-1)%n] + red[i,j]*red[(i+1)%n,j] + red[i,j]*red[(i-1)%n,j])
        if (DE <= 0):
            E += DE
            M += -2*red[i,j]
            red[i,j] = -red[i,j]
        else:
            p = np.random.rand()
            if(p<np.exp(-DE)):
                E += DE
                M += -2*red[i,j]
                red[i,j] = -red[i,j]
        En.append(E)
        Mn.append(M)
    
    #Se muestran gráficas cada 10 iteraciones
    if (counter%10 == 0):        
        fig, ax = plt.subplots(figsize=(6,4),tight_layout=True)
        ax.plot(range(S),En)
    
        ax.set_title(r'$T$ = 'f'{T}')
        ax.set_ylabel(r'$E$')
        ax.set_xlabel(r'$t$')
    
        plt.show()
    
        fig, ax = plt.subplots(figsize=(6,4),tight_layout=True)
        ax.plot(range(S),Mn)
    
        ax.set_title(r'$T$ = 'f'{T}')
        ax.set_ylabel(r'$M$')
        ax.set_xlabel(r'$t$')
    
        plt.show()
    
    #Se guardan los promedios en los arreglos
    E_prom.append(np.mean(En[int(S/5):]))
    M_prom.append(np.mean(Mn[int(S/5):]))
    E_std.append(np.std(En[int(S/5):]))
    M_std.append(np.std(Mn[int(S/5):]))
    
    counter += 1
    print('Completado: ', counter*100/n_T, '%')
    
#Temperatura crítica en la solución exacta
def f_aux(x):
    return np.sinh(2/x)-1

T_c = root(f_aux,2.2).x[0]

#Magnetización en la solución exacta de Onsager
def M_Ons(T_vals):
    Mag = []
    for T in T_vals:
        if(T<T_c):
            Mag.append((1-(np.sinh(2/T))**(-4))**(1/8))
        else:
            Mag.append(0)
    return Mag


#Energía en la solución exacta de Onsager
def E_Ons(T_vals):
    L_vals = 1/T_vals
    R1 = -2*np.tanh(2*L_vals)
    R2 = (1-np.sinh(2*L_vals)**2)/(np.sinh(2*L_vals)*np.cosh(2*L_vals))
    R3 = (2*ellipk((2*np.sinh(2*L_vals)/(np.cosh(2*L_vals)**2))**2)/np.pi)-1
    return R1 + R2*R3

#Se grafican la magnetización y la energía por espín
plt.plot(T_vals,M_Ons(T_vals), color='blue', label='Sol. de Onsager', lw=1)
plt.plot(T_vals,abs(np.asarray(M_prom))/(n*n), color='red', label='Simulación', lw=1)
plt.legend()
plt.xlabel('$k_BT/J$')
plt.ylabel('$m$')
plt.savefig('Mag_meta.png', format='png', dpi=900)
plt.show()

plt.plot(T_vals,E_Ons(T_vals), color='blue', label='Sol. de Onsager', lw=1)
plt.plot(T_vals,(np.asarray(E_prom)*T_vals)/(n*n), color='red', label='Simulación', lw=1)
plt.legend()
plt.xlabel('$k_BT/J$')
plt.ylabel(r'$\varepsilon/J$')
plt.savefig('En_meta.png', format='png', dpi=900)
plt.show()

#Corrección de simulaciones con estados meta-estables
def corr(m):
    L = 1/T_vals[m]
    
    red = np.random.choice([-1,1],[n,n])
    E = E_in(red,L)
    M = np.sum(red)
    En = []
    Mn = []

    #Evolución utilizando el algoritmo
    for t in range(S):
        i = np.random.randint(n)
        j = np.random.randint(n)
        DE = 2*L*(red[i,j]*red[i,(j+1)%n] + red[i,j]*red[i,(j-1)%n] + red[i,j]*red[(i+1)%n,j] + red[i,j]*red[(i-1)%n,j])
        if (DE <= 0):
            E += DE
            M += -2*red[i,j]
            red[i,j] = -red[i,j]
        else:
            p = np.random.rand()
            if(p<np.exp(-DE)):
                E += DE
                M += -2*red[i,j]
                red[i,j] = -red[i,j]
        En.append(E)
        Mn.append(M)

    fig, ax = plt.subplots(figsize=(6,4),tight_layout=True)
    ax.plot(range(S),np.asarray(En)*T_vals[m]/(n*n))
    
    ax.set_title(r'$k_BT/J$ = 'f'{T_vals[m]}')
    ax.set_ylabel(r'$\varepsilon/J$')
    ax.set_xlabel(r'$t$')
    plt.savefig(f'{m}''M.png', format='png', dpi=900)
    plt.show()
    
    fig, ax = plt.subplots(figsize=(6,4),tight_layout=True)
    ax.plot(range(S),np.asarray(Mn)/(n*n))
    
    ax.set_title(r'$k_BT/J$ = 'f'{T_vals[m]}')
    ax.set_ylabel(r'$m$')
    ax.set_xlabel(r'$t$')
    plt.savefig(f'{m}''E.png', format='png', dpi=900)
    plt.show()
    
    cut = float(input('Fracción a descartar '))
    E_prom[m] = np.mean(En[int(S*cut):])
    M_prom[m] = np.mean(Mn[int(S*cut):])
    #E_std[m] = np.std(En[int(S*cut):])
    #M_std[m] = np.std(Mn[int(S*cut):])


#Gráficas en función de la temperatura tras la corrección
plt.plot(T_vals,M_Ons(T_vals), color='blue', label='Sol. de Onsager', lw=1)
plt.plot(T_vals,abs(np.asarray(M_prom))/(n*n), color='red', label='Simulación', lw=1)
plt.legend()
plt.xlabel('$k_BT/J$')
plt.ylabel('$m$')
plt.savefig('Mag_final.png', format='png', dpi=900)
plt.show()

plt.plot(T_vals,E_Ons(T_vals), color='blue', label='Sol. de Onsager', lw=1)
plt.plot(T_vals,(np.asarray(E_prom)*T_vals)/(n*n), color='red', label='Simulación', lw=1)
plt.legend()
plt.xlabel('$k_BT/J$')
plt.ylabel(r'$\varepsilon/J$')
plt.savefig('En_final.png', format='png', dpi=900)
plt.show()



