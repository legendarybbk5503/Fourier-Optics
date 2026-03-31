import numpy as np
import numpy.typing as npt
from fourier import fourier
from matplotlib import pyplot as plt
from matplotlib import cm

class slit_2d:
    def __init__(self, N2, M2):
        self.__N: int = N2 // 2
        self.__M: int = M2 // 2
        self.__tn: npt.NDArray[np.integer] = np.arange(-self.__N, self.__N)
        self.__tm: npt.NDArray[np.integer] = np.arange(-self.__M, self.__M)
        
    def singleslit(self, **kwargs):
        fxy = kwargs.get("fx", False)
        real = kwargs.get("real", False)
        imag = kwargs.get("imag", False)
        mag = kwargs.get("mag", False)
        
        tn = self.__tn
        tm = self.__tm
        x = np.zeros((2*self.__N, 2*self.__M), dtype=np.complex128)
        x[np.ix_([i+self.__N for i in tn if -1 <= i <= 1], [j+self.__M for j in tm if -10 <= j <= 10])] = 1
        y = fourier().ft2(x)
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        U, V = np.meshgrid(tn, tm, indexing="ij")
        if fxy: surf = ax.plot_surface(U, V, x, cmap=cm.coolwarm)
        if real: surf =  ax.plot_surface(U, V, y.real, cmap=cm.coolwarm)
        if imag: surf = ax.plot_surface(U, V, y.imag, cmap=cm.coolwarm)
        if mag: surf = ax.plot_surface(U, V, np.abs(y), cmap=cm.coolwarm)
        ax.set_title("2D Single Slit")
        ax.set_xlabel('u')
        ax.set_ylabel('v')
        if real: ax.set_zlabel("Re(F(u, v))")
        if mag: ax.set_zlabel("|F(u, v)|")
        fig.colorbar(surf)
        

    def doubleslit(self, **kwargs):
        fxy = kwargs.get("fx", False)
        real = kwargs.get("real", False)
        imag = kwargs.get("imag", False)
        mag = kwargs.get("mag", False)
        
        tn = self.__tn
        tm = self.__tm
        x = np.zeros((2*self.__N, 2*self.__M), dtype=np.complex128)
        x[np.ix_([i+self.__N for i in tn if (-11 <= i <= -9) or (9 <= i <= 11)], [j+self.__M for j in tm if -10 <= j <= 10])] = 1
        y = fourier().ft2(x)
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        U, V = np.meshgrid(tn, tm, indexing="ij")
        if fxy: surf = ax.plot_surface(U, V, x, cmap=cm.coolwarm)
        if real: surf = ax.plot_surface(U, V, y.real, cmap=cm.coolwarm)
        if imag: surf = ax.plot_surface(U, V, y.imag, cmap=cm.coolwarm)
        if mag: surf = ax.plot_surface(U, V, np.abs(y), cmap=cm.coolwarm)
        ax.set_title("2D Double Slit")
        ax.set_xlabel('u')
        ax.set_ylabel('v')
        if real: ax.set_zlabel("Re(F(u, v))")
        if mag: ax.set_zlabel("|F(u, v)|")
        fig.colorbar(surf)

if __name__ == "__main__":
    dft = slit_2d(50, 50)
    #dft.singleslit(real=False, mag=True)
    dft.doubleslit(real=False, mag=True)

    plt.show()