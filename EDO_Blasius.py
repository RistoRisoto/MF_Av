import numpy as np
import matplotlib.pyplot as plt

# f''' + f*f'' = 0
#eta = 0: f=f'=0, f''= f''0; f0'' / lim(eta->inf)f'=1 
def sistema_edo(eta, F):
    f1, f2, f3 = F
    f1_prima = f2
    f2_prima = f3
    f3_prima = -f1*f3
    return np.array([f1_prima, f2_prima, f3_prima])

def rk4_vectorial(F, y0, eta_span, h=10e-3):
    eta0, etaf = eta_span
    eta = np.arange(eta0, etaf + h, h)
    
    # Matriz para guardar el estado [y1, y2] en cada paso de tiempo
    Y = np.zeros((len(eta), len(y0)))
    Y[0] = y0
    
    for i in range(len(eta) - 1):
        etai = eta[i]
        Yi = Y[i]
        
        k1 = F(etai, Yi)
        k2 = F(etai + 0.5 * h, Yi + 0.5 * h * k1)
        k3 = F(etai + 0.5 * h, Yi + 0.5 * h * k2)
        k4 = F(etai + h, Yi + h * k3)
        
        Y[i+1] = Yi + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        
    return eta, Y

f0_2prima = [0.4696,0.5]

eta_span = [0,10]
for f0_2prima in f0_2prima:
    F0 = np.array([0,0,f0_2prima])

    eta, F = rk4_vectorial(sistema_edo, F0, eta_span)

    plt.plot(F[:,1],eta,'-',label=r'$f^{\prime \prime}_0 ='+str(f0_2prima)+ '$')
plt.grid()
plt.legend()
plt.xlabel('$f^{\prime}$')
plt.ylabel('$\eta$')
plt.title('Solución de placa plana incompresible sin gradiente de presión (Blasius)')
plt.show()

