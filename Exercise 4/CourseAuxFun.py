import numpy as np
import scipy as sp

from scipy.spatial.distance import cdist


class CMDS():
    def __init__(self, d: int = 2):
        '''
        Constructing the object.
        Args:
            d - Number of dimensions of the encoder output.
        '''
        # ===========================Fill This===========================#
        # 1. Keep the model parameters.

        self.d = d
        self.Σ_d = None
        self.v_d = None
        self.mDxx_row_mean = None
        self.encoder = None
        self.mDxx = None

        # ===============================================================#

    def fit(self, mDxx: np.ndarray):
        '''
        Fitting model parameters to the input.
        Args:
            mDxx - Input data (Distance matrix) with shape Nx x Nx.
        Output:
            self
        '''
        # ===========================Fill This===========================#
        # 1. Build the model encoder.

        mDxxJ = mDxx - np.mean(mDxx, axis=0).reshape(-1, 1)
        mJDxxJ = mDxxJ - np.mean(mDxxJ, axis=0).reshape(1, -1)
        mKxx_centered = -0.5 * mJDxxJ

        v_d, Σ_d_power2, _ = sp.sparse.linalg.svds(mKxx_centered, k=self.d)
        self.Σ_d = np.diag(np.sqrt(Σ_d_power2))
        self.v_d = v_d
        self.mDxx_row_mean = np.mean(mDxx, axis=1).reshape(-1,1)
        self.encoder = (self.Σ_d @ self.v_d.T).T
        self.mDxx = mDxx

        # ===============================================================#
        return self

    def transform(self, mDxy: np.ndarray) -> np.ndarray:
        '''
        Applies (Out of sample) encoding.
        Args:
            mDxy - Input data (Distance matrix) with shape Nx x Ny.
        Output:
            mZ - Low dimensional representation (embeddings) with shape Ny x d.
        '''
        # ===========================Fill This===========================#
        # 1. Encode data using the model encoder.

        k_xy_centered = mDxy - self.mDxx_row_mean
        k_xy_centered = -0.5 * (k_xy_centered - np.mean(k_xy_centered, axis=0).reshape(1, -1))

        mZ = (np.linalg.inv(self.Σ_d) @ self.v_d.T @ k_xy_centered).T

        # ===============================================================#

        return mZ

    def fit_transform(self, mDxx: np.ndarray) -> np.ndarray:
        '''
        Applies encoding on the input.
        Args:
            mDxx - Input data (Distance matrix) with shape Nx x Nx.
        Output:
            mZ - Low dimensional representation (embeddings) with shape Nx x d.
        '''
        # ===========================Fill This===========================#
        # 1. Encode data using the model encoder.

        self.fit(mDxx)
        mZ = self.encoder

        # ===============================================================#

        return mZ

    
class MMDS():
    def __init__(self, d: int = 2, maxIter=500, ε=1e-3):
        '''
        Constructing the object.
        Args:
            d       - Number of dimensions of the encoder output.
            maxIter - Maximum number of iterations for the Majorization Minimization.
            ε       - Convergence threshold.
        '''
        # ===========================Fill This===========================#
        # 1. Keep the model parameters.

        self.d = d
        self.ε = ε
        self.maxIter = maxIter
        self.mZ = None

        # ===============================================================#

    def fit(self, mDxx: np.ndarray):
        '''
        Fitting model parameters to the input.
        Args:
            mDxx - Input data (Distance matrix) with shape Nx x Nx.
        Output:
            self
        '''
        # ===========================Fill This===========================#
        # 1. Build the model encoder.

        Nx = mDxx.shape[0]
        mZ_t_next = np.random.rand(Nx, self.d)
        for i in range(self.maxIter):
            mZ_t = mZ_t_next.copy()
            mDzz = cdist(mZ_t, mZ_t)

            mC = np.zeros(mDzz.shape)
            mC[mDzz != 0] = -mDxx[mDzz != 0] / (mDzz[mDzz != 0])
            mC[mDzz == 0] = 0

            mB = mC - np.diag(np.sum(mC, axis=1))
            mZ_t_next = (1 / Nx) * mB @ mZ_t

            curr_dist = np.linalg.norm(mZ_t_next - mZ_t, ord='fro')
            if curr_dist <= self.ε:
                break

        self.mZ = mZ_t_next

        # ===============================================================#
        return self

    def fit_transform(self, mDxx: np.ndarray) -> np.ndarray:
        '''
        Applies encoding on input data.
        Args:
            mDxx - Input data (Distance matrix) with shape Nx x Nx.
        Output:
            mZ - Low dimensional representation (embeddings) with shape Nx x d.
        '''
        # ===========================Fill This===========================#
        # 1. Apply the `fit()` method.
        # 2. Applies the Majorization Minimization.
        # 3. Encode data using the model encoder.
        # !! Use no loops beside the main loop (`maxIter`).

        self.fit(mDxx)
        mZ = self.mZ

        # ===============================================================#
        return mZ