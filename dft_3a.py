import numpy as np
from fourier import fourier
from matplotlib import pyplot as plt

class single_slit:
    def __init__(self, N):
        self.__N = N
        self.__t = np.arange(-N, N)
        
    def centre_width10(self, **kwargs):
        fx = kwargs.get("fx", False)
        real = kwargs.get("real", False)
        imag = kwargs.get("imag", False)
        mag = kwargs.get("mag", False)
        
        t = self.__t
        x = np.array([1 if -5<=i<=5 else 0 for i in t])
        y = fourier().ft1(x)
        
        if fx: plt.plot(t, x, label="x")
        if real: plt.plot(t, y.real, label="Re(F(u))")
        if imag: plt.plot(t, y.imag, label="Im(F(u))")
        if mag: plt.plot(t, np.abs(y), label="|F(u)|")
        plt.xlabel("u")
        plt.ylabel("F(u)")
        plt.legend()
        plt.grid(True)
        
    def shifted_width10(self, **kwargs):
        fx = kwargs.get("fx", False)
        real = kwargs.get("real", False)
        imag = kwargs.get("imag", False)
        mag = kwargs.get("mag", False)
        
        t = self.__t
        x = np.array([1 if -5-10<=i<=5-10 else 0 for i in t])
        y = fourier().ft1(x)
        
        if fx: plt.plot(t, x, label="x")
        if real: plt.plot(t, y.real, label="Re(F(u))")
        if imag: plt.plot(t, y.imag, label="Im(F(u))")
        if mag: plt.plot(t, np.abs(y), label="|F(u)|")
        plt.xlabel("u")
        plt.ylabel("F(u)")
        plt.legend()
        plt.grid(True)
    
    def centre_width20(self, **kwargs):
        fx = kwargs.get("fx", False)
        real = kwargs.get("real", False)
        imag = kwargs.get("imag", False)
        mag = kwargs.get("mag", False)
        
        t = self.__t
        x = np.array([1 if -10<=i<=10 else 0 for i in t])
        y = fourier().ft1(x)
        
        if fx: plt.plot(t, x, label="x")
        if real: plt.plot(t, y.real, label="Re(F(u))")
        if imag: plt.plot(t, y.imag, label="Im(F(u))")
        if mag: plt.plot(t, np.abs(y), label="|F(u)|")
        plt.ylabel("F(u)")
        plt.legend()
        plt.grid(True)
        
    def centre_width20_heighthalf(self, **kwargs):
        fx = kwargs.get("fx", False)
        real = kwargs.get("real", False)
        imag = kwargs.get("imag", False)
        mag = kwargs.get("mag", False)
        
        t = self.__t
        x = np.array([.5 if -10<=i<=10 else 0 for i in t])
        y = fourier().ft1(x)
        
        if fx: plt.plot(t, x, label="x")
        if real: plt.plot(t, y.real, label="Re(F(u))")
        if imag: plt.plot(t, y.imag, label="Im(F(u))")
        if mag: plt.plot(t, np.abs(y), label="|F(u)|")
        plt.xlabel("u")
        plt.ylabel("F(u)")
        plt.legend()
        plt.grid(True)

if __name__ == "__main__":
    dft = single_slit(250)
    #dft.centre_width10(real=False, mag=True)
    #dft.shifted_width10(real=False, mag=True)
    dft.centre_width20(real=False, mag=True)
    dft.centre_width20_heighthalf(mag=True)
    #plt.title("Height 1 vs Height 0.5 (Width 20)")
    plt.ylabel("|F(u)|")
    plt.show()