import numpy as np
import numpy.typing as npt

class fourier:
    def __init__(self):
        pass
    
    def ft1(self, X:npt.NDArray[np.complexfloating])->npt.NDArray[np.complexfloating]:
        """
        Author: Bosco Kwong
        Date: 8/11/2025
        Summary:
            Fourier transform function for a 1-dimensional signal

        Args:
            X (npt.NDArray[np.complexfloating]): a column vector of signal with N complex elements

        Returns:
            npt.NDArray[np.complexfloating]: the Fourier transfored signal in the form of a column vector of N complex elements.
        
        Example use:
            t = np.arange(-N, N)
            x = np.array([1 if -10<=i<=10 else 0 for i in t])
            y = fourier().ft1(x)
        """
        X_size: int = X.size
        N: int= X_size // 2
        t: npt.NDArray[np.integer] = np.arange(-N, N)
        
        y: npt.NDArray[np.complexfloating] = np.zeros(X_size, dtype=np.complexfloating256)
        for i, u in enumerate(t):
            y[i] = sum(f * np.exp(-np.pi*1j*x*u/N) for x, f in zip(t, X))
        return y
    
    def inverse_ft1(self, X:npt.NDArray[np.complexfloating])->npt.NDArray[np.complexfloating]:
        """
        Author: Bosco Kwong
        Date: 8/11/2025
        Summary:
            Inverse Fourier transform function for a 1-dimensional signal

        Args:
            X (npt.NDArray[np.complexfloating]): a column vector of signal with N complex elements

        Returns:
            npt.NDArray[np.complexfloating]: the inverse Fourier transfored signal in the form of a column vector of N complex elements.
        
        Example use:
            t = np.arange(-N, N)
            x = np.array([1 if -10<=i<=10 else 0 for i in t])
            y = fourier().inverse_ft1(x)
        """
        X_size: int = X.size
        N: int= X_size // 2
        t: npt.NDArray[np.integer] = np.arange(-N, N)
        
        y: npt.NDArray[np.complexfloating] = np.zeros(X_size, dtype=np.complexfloating)
        for i, u in enumerate(t):
            y[i] = sum(f * np.exp(np.pi*1j*x*u/N) for x, f in zip(t, X))
        return y
    
    def convolution1(self, u:npt.NDArray[np.complexfloating], v:npt.NDArray[np.complexfloating]) -> npt.NDArray[np.complexfloating]:
        return self.inverse_ft1(self.ft1(u)*self.ft1(v)) / u.size
    
    def ft2(self, X:npt.NDArray[np.complexfloating])->npt.NDArray[np.complexfloating]:
        """
        Author: Bosco Kwong
        Date: 8/11/2025
        Fourier transform function for 2-dimensional signal.

        Args:
            X (npt.NDArray[np.complexfloating]): a matrix of 2D signal with the size (2M x 2N).

        Returns:
            npt.NDArray[np.complexfloating]:the Fourier transformed signal in the form of a (2M x 2N) matrix with complex elements.
        
        Example use:
            x = np.zeros((2*self.__N, 2*self.__M), dtype=np.complex128)
            x[10][10]=1
            y = fourier().ft2(x)
        """
        N: int
        M: int
        row, column = X.shape
        N = row // 2
        M = column // 2
        tn: npt.NDArray[np.integer] = np.arange(-N, N)
        tm: npt.NDArray[np.integer] = np.arange(-M, M)
        Y: npt.NDArray[np.complexfloating] = np.zeros(X.shape, dtype=np.complexfloating)
        
        for i, u in enumerate(tn):
            for j, v in enumerate(tm):
                Y[i][j] = sum(X[x+N][y+M] * np.exp(-np.pi*1j*(u*x/N+v*y/M)) for x in tn for y in tm)
        Y /= (4*N*M)
        return Y