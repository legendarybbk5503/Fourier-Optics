from dft_3a import single_slit
from dft_3b import double_slit
from dft_3c import convolution
from dft_4 import slit_2d
from matplotlib import pyplot as plt

def main():
    dft = single_slit(250)
    dft.centre_width10(real=True)
    plt.show()

if __name__ == "__main__":
    main()