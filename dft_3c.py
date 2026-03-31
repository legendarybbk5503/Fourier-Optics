import numpy as np
from fourier import fourier
from matplotlib import pyplot as plt
from dft_3a import single_slit

class convolution:
    def __init__(self, N):
        self.__N = N
        self.__t = np.arange(-N, N)
        
    def singleslit(self, **kwargs):
        fx = kwargs.get("fx", False)
        real = kwargs.get("real", False)
        imag = kwargs.get("imag", False)
        mag = kwargs.get("mag", False)
        
        t = self.__t
        x = np.array([1 if -10<=i<=10 else 0 for i in t])
        y = fourier().convolution1(x, x)
        
        if fx: plt.plot(t, x, label="x")
        if real: plt.plot(t, y.real, label="Re(h(x))")
        if imag: plt.plot(t, y.imag, label="Im(h(x))")
        if mag: plt.plot(t, np.abs(y), label="|h(x)|")
        plt.xlabel("x")
        plt.ylabel("h(x)")
        plt.legend()
        plt.grid(True)
    
    def singleslit_check(self, **kwargs):
        fx = kwargs.get("fx", False)
        real = kwargs.get("real", False)
        imag = kwargs.get("imag", False)
        mag = kwargs.get("mag", False)
        
        t = self.__t
        x = np.array([1 if -10<=i<=10 else 0 for i in t])
        h = fourier().convolution1(x, x)
        H = fourier().ft1(h)
        F2 = (fourier().ft1(x))**2
        
        if fx:
            plt.plot(t, x, label="x")
            plt.plot(t, h, label="h")
        if real:
            plt.plot(t, H.real, label="Re(H(u))")
            plt.plot(t, F2.real, label="Re(F^2(u))")
        if imag:
            plt.plot(t, H.imag, label="Im(H(u))")
            plt.plot(t, F2.imag, label="Im(F^2(u))")
        if mag:
            plt.plot(t, np.abs(H)-np.abs(F2), label="|H(u)|")
            #plt.plot(t, np.abs(F2), label="|F^2(u)|")
        plt.xlabel("u")
        plt.ylabel("F^2(u), H(u)")
        plt.legend()
        plt.grid(True)

if __name__ == "__main__":
    con = convolution(250)
    #con.singleslit(mag=True)
    con.singleslit_check(mag=True)
    plt.title("Verification of Convolution Theorem")
    plt.show()