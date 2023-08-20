import numpy as np
import matplotlib.pyplot as plt

"""
H. Guo, "A Simple Algorithm for Fitting a Gaussian Function [DSP Tips and Tricks],
" in IEEE Signal Processing Magazine, vol. 28, no. 5, pp. 134-137, Sept. 2011, doi: 10.1109/MSP.2011.941846.
"""

def gaussian1D(x, A=1, mu=0.2, sigma=3):
    """
    y = Ae^(-(x-mu)^2/2*sigma^2)
    See Eq. (1)
    x: time axis or samples
    A: Amplitude
    mu: mean value
    sigma: standard deviation
    return y: Amplitude of given gaussian for all x
    """
    y = A * np.exp((-(x - mu) ** 2) / (2 * sigma ** 2))
    return y


def parabola1D(x, a, b, c):
    """
    See Eq. (2)
    x: time axis or samples
    a,b,c polynomial parameters to describe a parabolic function
    return y: Amplitude of given parabola for all x
    """
    y = c * x ** 2 + b * x + a
    return y


def get_params(A=1, mu=0.2, sigma=3):
    """
    Calculate parabola coefficients a-c in log space.
    See Eq. (2)
    """
    A = A
    a = np.log(A) - (mu ** 2 / (2 * sigma ** 2))
    b = (mu / sigma ** 2)  # * x
    c = (-1 / (2 * sigma ** 2))  # * x**2
    return a, b, c


def get_gauss_params(a, b, c):
    """
    Calculate gaussian coefficients from parabola coefficients.
    See Eqs. (5-7)
    """
    mu = -b / (2 * c)
    sigma = np.sqrt((-1 / (2 * c)))
    A = np.exp(((a - b ** 2) / 4 * c))

    return A, mu, sigma


def least_squares_fit(Mat, Z):
    """
    Solve y=Xa using least squares method
    lsq. fit of given matrix X for given a
    Mat: Coefficient matrix X
    Z: Solution vector a
    return y: Minimized parameters of y
    """
    # y = Xa --> a = (X^T@X)^-1@a
    # https://mathworld.wolfram.com/LeastSquaresFittingPolynomial.html
    # https://math.stackexchange.com/questions/481845/regression-with-a-vandermonde-matrix
    # Moore Penrose pseudoinverse
    MP_inv = np.linalg.pinv(Mat.T @ Mat) @ Mat.T
    res = MP_inv @ Z

    return res


def getMat(y, x):
    """
    Calculate Matrix fro msamples x,y given by Eq. (15)
    """
    M = np.array([[sum(y * y), sum(x * y * y), sum(x * x * y * y)],
                  [sum(x * y * y), sum(x * x * y * y), sum(x * x * x * y * y)],
                  [sum(x * x * y * y), sum(x * x * x * y * y), sum(x * x * x * x * y * y)]])

    return M


def getZ(y, x):
    """
    calculate solution vector from samples x,y given by Eq. (15)
    """
    z = np.array([[sum(y * y * np.log(y))],
                  [sum(y * y * x * np.log(y))],
                  [sum(x * x * y * y * np.log(y))]])
    return z


def filter_data(y, x, tresh=0.1):
    """
    Filter samples below given treshhold as suggested in subsection effects of noise
    """
    filt = np.argwhere(y < tresh)
    y_new = np.delete(y, filt)
    x_new = np.delete(x, filt)

    return y_new, x_new


if __name__ == "__main__":
    N = 350  # Number of samples
    xin1d = np.linspace(-10, 10, N)  # Generate x values for 1D gauss
    data1D = gaussian1D(xin1d) + np.random.random(N)/10   # Add some normal distributed noise

    # Reconstruct gauss using the method as suggested in subsection "Weighted Least Squares Estimation"
    mat = getMat(data1D, xin1d)
    z = getZ(data1D, xin1d)
    res = least_squares_fit(mat, z)
    a = res[0]
    b = res[1]
    c = res[2]

    print(a, b, c)
    A, mu, sigma = get_gauss_params(a, b, c)
    a_org, b_org, c_org = get_params()
    A_org, mu_org, sigma_org = get_gauss_params(a_org, b_org, c_org)
    print(a_org, b_org, c_org)

    print(A, mu, sigma)
    print(A_org, mu_org, sigma_org)

    y_parab = parabola1D(xin1d, a, b, c)
    y_gaussian = gaussian1D(xin1d, A, mu, sigma)
    plt.figure(2)
    plt.scatter(xin1d, data1D, label='Raw samples')
    plt.plot(xin1d, y_parab, c='g', label='Fitted parabola')
    plt.plot(xin1d, y_gaussian, c='r', label='Fitted gauss')
    plt.xlabel('Time/ Samples along x')
    plt.ylabel('Gaussian amplitude A ')
    plt.legend()
    plt.show()

    """
    Redo it with filtering of small amplitude values 
    """

    yfilt, xfilt = filter_data(data1D, xin1d, 0.05)

    mat = getMat(yfilt, xfilt)
    z = getZ(yfilt, xfilt)
    res = least_squares_fit(mat, z)
    afilt = res[0]
    bfilt = res[1]
    cfilt = res[2]
    print(afilt, bfilt, cfilt)
    print(a_org, b_org, c_org)
    Afilt, mufilt, sigmafilt = get_gauss_params(afilt, bfilt, cfilt)

    print(Afilt, mufilt, sigmafilt)
    print(A_org, mu_org, sigma_org)

    y_parab_filt = parabola1D(xfilt, afilt, bfilt, cfilt)
    y_gaussian_filt = gaussian1D(xfilt, Afilt, mufilt, sigmafilt)
    plt.figure(2)
    plt.scatter(xfilt, yfilt, label='Raw samples')
    plt.plot(xfilt, y_parab_filt, c='g', label='Fitted parabola')
    plt.plot(xfilt, y_gaussian_filt, c='r', label='Fitted gauss')
    plt.xlabel('Time/ Samples along x')
    plt.ylabel('Gaussian amplitude A ')
    plt.legend()
    plt.show()
