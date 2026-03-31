import numpy as np
from fourier import fourier
from matplotlib import pyplot as plt

class double_slit:
    def __init__(self, N):
        self.__N = N
        self.__t = np.arange(-N, N)
        
    def at_pm15(self, **kwargs):
        fx = kwargs.get("fx", False)
        real = kwargs.get("real", False)
        imag = kwargs.get("imag", False)
        mag = kwargs.get("mag", False)
        
        t = self.__t
        x = np.array([1 if -15-10<=i<=-15+10 or 15-10<=i<=15+10 else 0 for i in t])
        y = fourier().ft1(x)
        
        if fx: plt.plot(t, x, label="x")
        if real: plt.plot(t, y.real, label="Re(F(u))")
        if imag: plt.plot(t, y.imag, label="Im(F(u))")
        if mag: plt.plot(t, np.abs(y), label="|F(u)|")
        plt.xlabel("u")
        plt.ylabel("F(u)")
        plt.legend()
        plt.grid(True)

    def at_pm25(self, **kwargs):
        fx = kwargs.get("fx", False)
        real = kwargs.get("real", False)
        imag = kwargs.get("imag", False)
        mag = kwargs.get("mag", False)
        
        t = self.__t
        x = np.array([1 if -25-10<=i<=-25+10 or 25-10<=i<=25+10 else 0 for i in t])
        y = fourier().ft1(x)
        
        if fx: plt.plot(t, x, label="x")
        if real: plt.plot(t, y.real, label="Re(F(u))")
        if imag: plt.plot(t, y.imag, label="Im(F(u))")
        if mag: plt.plot(t, np.abs(y), label="|F(u)|")
        plt.xlabel("u")
        plt.ylabel("F(u)")
        plt.legend()
        plt.grid(True)

    def at_pm25_doublewidth(self, **kwargs):
        fx = kwargs.get("fx", False)
        real = kwargs.get("real", False)
        imag = kwargs.get("imag", False)
        mag = kwargs.get("mag", False)
        
        t = self.__t
        x = np.array([1 if -25-20<=i<=-25+20 or 25-20<=i<=25+20 else 0 for i in t])
        y = fourier().ft1(x)
        
        if fx: plt.plot(t, x, label="x")
        if real: plt.plot(t, y.real, label="Re(F(u))")
        if imag: plt.plot(t, y.imag, label="Im(F(u))")
        if mag: plt.plot(t, np.abs(y), label="|F(u)|")
        plt.xlabel("u")
        plt.ylabel("F(u)")
        plt.legend()
        plt.grid(True)
        
    def at_pm25_halfheight(self, **kwargs):
        fx = kwargs.get("fx", False)
        real = kwargs.get("real", False)
        imag = kwargs.get("imag", False)
        mag = kwargs.get("mag", False)
        
        t = self.__t
        x = np.array([.5 if -25-10<=i<=-25+10 or 25-10<=i<=25+10 else 0 for i in t])
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
    dft = double_slit(250)
    dft.at_pm15(fx=False, real=False, mag=True)
    #dft.at_pm25(fx=False, real=False, mag=True)
    #dft.at_pm25_doublewidth(fx=False, real=False, mag=True)
    #dft.at_pm25_halfheight(mag=True)
    plt.title("Double Slit at +-15")
    plt.show()