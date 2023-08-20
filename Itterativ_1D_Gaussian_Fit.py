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
    z = A * np.exp((-(x - mu) ** 2) / (2 * sigma ** 2))
    return z


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
    Calculate Matrix fro msamples x,y given by Eq. (16)
    """
    M = np.array([[sum(y * y), sum(x * y * y), sum(x * x * y * y)],
                  [sum(x * y * y), sum(x * x * y * y), sum(x * x * x * y * y)],
                  [sum(x * x * y * y), sum(x * x * x * y * y), sum(x * x * x * x * y * y)]])

    return M


def getZ(y, x, y_hat):
    """
    calculate solution vector from samples x,y given by Eq. (16)
    """
    z = np.array([[sum(y * y * np.log(y_hat))],
                  [sum(y * y * x * np.log(y_hat))],
                  [sum(x * x * y * y * np.log(y_hat))]])
    return z


def filter_data(y, x, tresh=0.1):
    """
    Filter samples below given treshhold as suggested in subsection effects of noise
    """
    filt = np.argwhere(y < tresh)
    y_new = np.delete(y, filt)
    x_new = np.delete(x, filt)

    return y_new, x_new


def calc_y_prev(x, a, b, c):
    """
    Calc previouse y values as described in Eq. (17) for k>0
    """
    ytick = np.exp(a + x * b + x * x * c)

    return ytick


def itterative_fit(data_points, sample_length):
    """
    params:
    data_points: number of total samples in individual column
    sample_length: length of buffered samples length of subsets
    """
    subsets = int(len(data_points)) / sample_length
    i = 0
    x = data_points[i * sample_length:(i + 1) * sample_length]
    y = data_points[i * sample_length:(i + 1) * sample_length]
    mat = getMat(y, x)
    z = getZ(y, x, y)
    res = least_squares_fit(mat, z)
    a = res[0]
    b = res[1]
    c = res[2]

    a_itt = []
    a_itt.append(a)
    b_itt = []
    b_itt.append(b)
    c_itt = []
    c_itt.append(c)
    for i in range(int(subsets) - 2):
        x = xin1d[(i + 1) * Nloc:(i + 2) * Nloc]
        y = calc_y_prev(x, a, b, c)
        y_hat = data1D[(i + 1) * Nloc:(i + 2) * Nloc]
        mat = getMat(y, x)
        z = getZ(y, x, y_hat)
        res = least_squares_fit(mat, z)
        a = res[0]
        b = res[1]
        c = res[2]
        a_itt.append(a)
        b_itt.append(b)
        c_itt.append(c)

        print(res)

    A, mu, sigma = get_gauss_params(a_itt[-1], b_itt[-1], c_itt[-1])
    y_gaussian = gaussian1D(xin1d, A, mu, sigma)

    return A, mu, sigma, y_gaussian


if __name__ == "__main__":

    # 1. generate data
    Ntot = 1000  # Number of samples
    Nloc = 100  # Subsample length
    xin1d = np.linspace(-10, 10, Ntot)  # Generate x values for 1D gauss
    data1D = gaussian1D(xin1d) + np.random.random(Ntot) / 100

    subsets = Ntot / Nloc  # Number of subsamples
    k = 0  # k = 0 See Eq. (17)
    x = xin1d[k * Nloc:(k + 1) * Nloc]
    y = data1D[k * Nloc:(k + 1) * Nloc]
    mat = getMat(y, x)
    z = getZ(y, x, y)
    res = least_squares_fit(mat, z)
    a = res[0]
    b = res[1]
    c = res[2]

    a_itt = [a]
    b_itt = [b]
    c_itt = [c]
    for k in range(int(subsets) - 2):
        x = xin1d[(k + 1) * Nloc:(k + 2) * Nloc]
        y = calc_y_prev(x, a, b, c)
        y_hat = data1D[(k + 1) * Nloc:(k + 2) * Nloc]
        mat = getMat(y, x)
        z = getZ(y, x, y_hat)
        res = least_squares_fit(mat, z)
        a = res[0]
        b = res[1]
        c = res[2]
        a_itt.append(a)
        b_itt.append(b)
        c_itt.append(c)

        print(res)

    plt.figure(1)
    plt.scatter(xin1d, data1D, label='Raw samples')

    err_itt = []
    mu_itt = []
    sigma_itt = []
    A_itt = []
    for k in range(len(a_itt)):
        A, mu, sigma = get_gauss_params(a_itt[k], b_itt[k], c_itt[k])
        y_gaussian = gaussian1D(xin1d, A, mu, sigma)
        print(A, mu, sigma)
        err_itt.append(abs(mu - 0.2))
        A_itt.append(A)
        mu_itt.append(mu)
        sigma_itt.append(sigma)
        plt.plot(xin1d, y_gaussian, label='Itt. ' + str(k))
    plt.legend()
    plt.xlabel('Time/ Samples along x')
    plt.ylabel('Gaussian amplitude A ')
    plt.show()

    A, mu, sigma = get_gauss_params(a_itt[-1], b_itt[-1], c_itt[-1])
    y_gaussian = gaussian1D(xin1d, A, mu, sigma)
    print(A, mu, sigma)

    plt.figure(2)
    plt.scatter(xin1d, data1D, label='Raw samples')
    plt.plot(xin1d, y_gaussian, c='r', label='Fitted gauss')
    plt.legend()
    plt.xlabel('Time/ Samples along x')
    plt.ylabel('Gaussian amplitude A ')
    plt.title('Final result')
    plt.show()

    plt.figure(3)
    plt.plot(A_itt, label='A')
    plt.plot(mu_itt, label='mu')
    plt.plot(sigma_itt, label='sigma')
    plt.plot(err_itt, label='Error')
    plt.legend()
    plt.xlabel('Itteration k')
    plt.ylabel('Param values ')
    plt.title('Evolving of gauss params over k')
    plt.show()
