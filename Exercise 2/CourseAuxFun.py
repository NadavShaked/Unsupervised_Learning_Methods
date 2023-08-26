import time
import numpy as np
from scipy.stats import multivariate_normal as MVN


def k_means_plus_plus_init(mX, K, seedNum):
    np.random.seed(seedNum)
    centroids = []
    new_centroid = np.array(mX[np.random.choice(mX.shape[0])])
    centroids.append(new_centroid.tolist())
    m_distance = None
    for k in range(1, K):
        if m_distance is not None:
            m_distance = np.concatenate((m_distance, np.linalg.norm(mX - new_centroid, axis=1).reshape(1, -1)))
        else:
            m_distance = np.linalg.norm(mX - new_centroid, axis=1).reshape(1, -1)
        distances_min = np.min(m_distance, axis=0)
        probs = distances_min / np.sum(distances_min)
        index = np.random.choice(mX.shape[0], p=probs)
        new_centroid = mX[index]
        centroids.append(new_centroid.tolist())

    return np.array(centroids)


def InitKMeans(mX: np.ndarray, K: int, initMethod: int = 0, seedNum: int = 123) -> np.ndarray:
    '''
    K-Means algorithm initialization.
    Args:
        mX          - Input data with shape N x d.
        K           - Number of clusters.
        initMethod  - Initialization method: 0 - Random, 1 - K-Means++.
        seedNum     - Seed number used.
    Output:
        mC          - The initial centroids with shape K x d.
    Remarks:
        - Given the same parameters, including the `seedNum` the algorithm must be reproducible.
    '''

    np.random.seed(seedNum)
    if initMethod == 0:
        mC = mX[np.random.choice(range(len(mX)), size=K)]
    else:
        mC = k_means_plus_plus_init(mX, K, seedNum)

    return mC


def CalcKMeansObj(mX: np.ndarray, mC: np.ndarray) -> float:
    '''
    K-Means algorithm.
    Args:
        mX          - The data with shape N x d.
        mC          - The centroids with shape K x d.
    Output:
        objVal      - The value of the objective function of the KMeans.
    Remarks:
        - The objective function uses the squared euclidean distance.
    '''

    m_distance = np.array([np.linalg.norm(mX - c, axis=1) for c in mC])
    return np.min(m_distance, axis=0).sum()


def KMeans(mX: np.ndarray, mC: np.ndarray, numIter: int = 1000, stopThr: float = 0) -> np.ndarray:
    '''
    K-Means algorithm.
    Args:
        mX          - Input data with shape N x d.
        mC          - The initial centroids with shape K x d.
        numIter     - Number of iterations.
        stopThr     - Stopping threshold.
    Output:
        mC          - The final centroids with shape K x d.
        vL          - The labels (0, 1, .., K - 1) per sample with shape (N, )
        lO          - The objective value function per iterations (List).
    Remarks:
        - The maximum number of iterations must be `numIter`.
        - If the objective value of the algorithm doesn't improve by at least `stopThr` the iterations should stop.
    '''

    i = 0
    diff_val_obj = np.inf
    prev_val_obj = np.inf
    lO = []
    while i < numIter and diff_val_obj > stopThr:
        print("iteration: " + str(i))
        lO.append(CalcKMeansObj(mX, mC))
        print(lO[-1])
        m_distance = np.array([np.linalg.norm(mX - c, axis=1) for c in mC])
        vL = np.argmin(m_distance, axis=0)
        mC = np.array([mX[np.where(vL == k)].mean(axis=0) for k in range(len(mC))])

        diff_val_obj = prev_val_obj - lO[-1]
        prev_val_obj = lO[-1]
        i += 1

    return mC, vL, lO


def InitGmm(mX: np.ndarray, K: int, seedNum: int = 123) -> np.ndarray:
    '''
    GMM algorithm initialization.
    Args:
        mX          - Input data with shape N x d.
        K           - Number of clusters.
        seedNum     - Seed number used.
    Output:
        mμ          - The initial mean vectors with shape K x d.
        tΣ          - The initial covariance matrices with shape (d x d x K).
        vW          - The initial weights of the GMM with shape K.
    Remarks:
        - Given the same parameters, including the `seedNum` the algorithm must be reproducible.
        - mμ Should be initialized by the K-Means++ algorithm.
    '''

    mμ = k_means_plus_plus_init(mX, K, seedNum)
    m_distance = np.array([np.linalg.norm(mX - c, axis=1) for c in mμ])
    vL = np.argmin(m_distance, axis=0)

    tΣ = np.zeros((mX.shape[1], mX.shape[1], K))
    for k in range(K):
        tΣ[:, :, k] = np.diag(np.var(mX[np.where(vL == k)], axis=0))

    vW = np.ones(K) / K

    return mμ, tΣ, vW


def CalcGmmObj(mX: np.ndarray, mμ: np.ndarray, tΣ: np.ndarray, vW: np.ndarray) -> float:
    '''
    GMM algorithm objective function.
    Args:
        mX          - The data with shape N x d.
        mμ          - The initial mean vectors with shape K x d.
        tΣ          - The initial covariance matrices with shape (d x d x K).
        vW          - The initial weights of the GMM with shape K.
    Output:
        objVal      - The value of the objective function of the GMM.
    Remarks:
        - A
    '''

    objVal = 0
    K = mμ.shape[0]
    N = mX.shape[0]

    for i in range(N):
        objVal += np.log(
            (vW * np.array([MVN.pdf(mX[i], mean=mμ[k], cov=tΣ[:, :, k], allow_singular=True) for k in range(K)])).sum())

    return objVal


def GMM(mX: np.ndarray, mμ: np.ndarray, tΣ: np.ndarray, vW: np.ndarray, numIter: int = 1000,
        stopThr: float = 1e-5) -> np.ndarray:
    '''
    GMM algorithm.
    Args:
        mX          - Input data with shape N x d.
        mμ          - The initial mean vectors with shape K x d.
        tΣ          - The initial covariance matrices with shape (d x d x K).
        vW          - The initial weights of the GMM with shape K.
        numIter     - Number of iterations.
        stopThr     - Stopping threshold.
    Output:
        mμ          - The final mean vectors with shape K x d.
        tΣ          - The final covariance matrices with shape (d x d x K).
        vW          - The final weights of the GMM with shape K.
        vL          - The labels (0, 1, .., K - 1) per sample with shape (N, )
        lO          - The objective function value per iterations (List).
    Remarks:
        - The maximum number of iterations must be `numIter`.
        - If the objective value of the algorithm doesn't improve by at least `stopThr` the iterations should stop.
    '''

    start_time = time.time()

    lO = []
    N, d = mX.shape
    K = mμ.shape[0]

    diff_val_obj = np.inf
    prev_val_obj = np.inf

    iter = 0
    while iter < numIter and diff_val_obj > stopThr:
        print("iteration: " + str(iter))
        lO.append(CalcGmmObj(mX, mμ, tΣ, vW))
        print(lO[-1])

        m_prob = np.array(
            [[(vW[k] * np.array(MVN.pdf(mX[i], mean=mμ[k], cov=tΣ[:, :, k], allow_singular=True))) for k in range(K)]
             for i in range(N)])
        p_x = m_prob / np.sum(m_prob, axis=1).reshape([-1, 1])

        vN = np.array([p_x[:, k].sum() for k in range(K)])

        vW = vN / N

        mμ = np.squeeze([(1 / vN[k] * (p_x[:, k].reshape(1, -1) @ mX)) for k in range(K)])

        tΣ = np.zeros((mX.shape[1], mX.shape[1], K))
        for k in range(K):
            s = 0
            for i in range(N):
                s += p_x[i, k] * (mX[i] - mμ[k]).reshape(d, -1) @ (mX[i] - mμ[k]).reshape(-1, d)
            tΣ[:, :, k] = (1 / vN[k]) * s

        diff_val_obj = np.abs(lO[-1] - prev_val_obj)
        prev_val_obj = lO[-1]
        iter += 1

    vL = np.array([np.random.choice(K, 1, p=p_x[i])[0] for i in range(N)])

    end_time = time.time()
    print(end_time - start_time)

    return mμ, tΣ, vW, vL, lO

def initKMeansForGMMHard(mX: np.ndarray, K: int, initMethod: int = 0, seedNum: int = 123) -> np.ndarray:
    '''
    K-Means algorithm initialization.
    Args:
        mX          - Input data with shape N x d.
        K           - Number of clusters.
        initMethod  - Initialization method: 0 - Random, 1 - K-Means++.
        seedNum     - Seed number used.
    Output:
        mC          - The initial centroids with shape K x d.
    Remarks:
        - Given the same parameters, including the `seedNum` the algorithm must be reproducible.
    '''
    np.random.seed(seedNum)
    N = mX.shape[0]
    d = mX.shape[1]
    if K > N:
        print(f"Error! maximum number of clusters is {N}, setting K={N}...")
        K = N
    shuffledDataset = np.random.permutation(N)
    if initMethod == 0:
        # permuting the input avoids the risk of selecting the same example twice
        centroids = mX[shuffledDataset[:K]]
    else:
        centroids = np.zeros((K, d))
        centroids[0] = mX[shuffledDataset[:1]]
        dist_mat = np.sum((centroids[0]-mX)**2, axis=1)
        for i in range(K-1):
            probs = dist_mat/np.sum(dist_mat)
            centroids[i] = mX[np.random.choice(N, p=probs)]
            dist_mat = np.minimum(dist_mat, np.sum((centroids[i]-mX)**2, axis=1))
    return centroids


def calcKMeansObjForGMMHard(mX: np.ndarray, mC: np.ndarray):
    '''
    K-Means algorithm.
    Args:
        mX          - The data with shape N x d.
        mC          - The centroids with shape K x d.
    Output:
        objVal      - The value of the objective function of the KMeans.
        pointsToCentroid - The centroid index (0, 1, .., K - 1) per sample with shape (N, )
        newCentroids - The new centroids with shape K x d.
    Remarks:
        - The objective function uses the squared euclidean distance.
    '''
    N = mX.shape[0]
    d = mX.shape[1]
    K = mC.shape[0]
    # assign point to centroid array
    pointsToCentroid = np.zeros(N, dtype=int)
    sumCentroids = np.zeros((K,d))
    newCentroids = np.zeros((K,d))
    squaredSum = 0.0
    
    # calculate distance between each point and each centroid
    dist = np.sum(mX**2, axis=1)[:, np.newaxis] + np.sum(mC**2, axis=1) - 2 * np.dot(mX, mC.T)
    
    # assign points to centroids
    pointsToCentroid = np.argmin(dist, axis=1)
    
    # update centroids
    for k in range(K):
        idx = (pointsToCentroid == k)
        mSum = np.sum(idx)
        if mSum == 0:
            continue
        newCentroids[k,:] = np.sum(mX[idx,:], axis=0)
        sumCentroids[k,:] = mSum
        
    # compute objective function
    squaredSum = np.sum(np.min(dist, axis=1))
    
    return squaredSum, pointsToCentroid, newCentroids / sumCentroids
   
def initGmmHard(mX: np.ndarray, K: int, seedNum: int = 123) -> np.ndarray:
    '''
    GMM algorithm initialization.
    Args:
        mX          - Input data with shape N x d.
        K           - Number of clusters.
        seedNum     - Seed number used.
    Output:
        mμ          - The initial mean vectors with shape K x d.
        tΣ          - The initial covariance matrices with shape (d x d x K).
        vW          - The initial weights of the GMM with shape K.
    Remarks:
        - Given the same parameters, including the `seedNum` the algorithm must be reproducible.
        - mμ Should be initialized by the K-Means++ algorithm.
    '''

    d = mX.shape[1]

    mu = initKMeansForGMMHard(mX, K, 1, seedNum)
    _, pointsToCentroid, __ = calcKMeansObjForGMMHard(mX, mu)
    vW = np.zeros(K)
    tSigma = np.zeros((d,d,K))

    for k in range(K):
        idx = (pointsToCentroid == k)
        mSum = np.sum(idx)
        if mSum == 0:
            continue
        vW[k] = 1/K
        tSigma[:,:,k] = np.diag(np.var(mX[idx], axis=0))
    return mu, tSigma, vW
    
    
def calcGmmHardObj(mX: np.ndarray, mu: np.ndarray, tSigma: np.ndarray, vW: np.ndarray) -> float:
    '''
    GMM algorithm objective function.
    Args:
        mX          - The data with shape N x d.
        mμ          - The initial mean vectors with shape K x d.
        tΣ          - The initial covariance matrices with shape (d x d x K).
        vW          - The initial weights of the GMM with shape K.
    Output:
        objVal      - The value of the objective function of the GMM.
    Remarks:
        - A
    '''

    K = mu.shape[0]
    d = mu.shape[1]
    N = mX.shape[0]
    pointsToCentroid = np.zeros(N)
    newCentroids = np.zeros((K,d))
    objVal = 0.0
    for i in range(N):
        kObjVal = 0.0
        bestResult = float('-inf')
        for k in range(K):
            currS = vW[k] * MVN.pdf(mX[i], mean=mu[k], cov=tSigma[:, :, k], allow_singular=True)
            if currS > bestResult:
                bestResult = currS
                pointsToCentroid[i] = k
            kObjVal += currS
        objVal += np.log(kObjVal)

    # update centroids
    for k in range(K):
        idx = np.squeeze((pointsToCentroid == k))
        mSum = np.sum(idx)
        newCentroids[k,:] = np.mean(mX[idx], axis=0)
        vW[k] = mSum/N
        tSigma[:, :, k] = np.cov(mX[idx].T)
    
    return objVal, pointsToCentroid, vW, newCentroids, tSigma


def GMMHard(mX: np.ndarray, mu: np.ndarray, tSigma: np.ndarray, vW: np.ndarray, numIter: int = 1000, stopThr: float = 1e-5) -> np.ndarray:
    '''
    GMM algorithm.
    Args:
        mX          - Input data with shape N x d.
        mμ          - The initial mean vectors with shape K x d.
        tΣ          - The initial covariance matrices with shape (d x d x K).
        vW          - The initial weights of the GMM with shape K.
        numIter     - Number of iterations.
        stopThr     - Stopping threshold.
    Output:
        mμ          - The final mean vectors with shape K x d.
        tΣ          - The final covariance matrices with shape (d x d x K).
        vW          - The final weights of the GMM with shape K.
        vL          - The labels (0, 1, .., K - 1) per sample with shape (N, )
        lO          - The objective function value per iterations (List).
    Remarks:
        - The maximum number of iterations must be `numIter`.
        - If the objective value of the algorithm doesn't improve by at least `stopThr` the iterations should stop.
    '''

    N = mX.shape[0]
    d = mX.shape[1]
    K = mu.shape[0]

    numIterations = 0
    lastObjectiveValue = 0.0
    currObjectiveValue, pointsToCentroid, vW, mu, tSigma = calcGmmHardObj(mX, mu, tSigma, vW)
    lO = [currObjectiveValue]
    while numIterations <= numIter and  abs(currObjectiveValue - lastObjectiveValue) > stopThr:
        lastObjectiveValue = currObjectiveValue
        numIterations += 1
        currObjectiveValue, pointsToCentroid, vW, mu, tSigma = calcGmmHardObj(mX, mu, tSigma, vW)
        lO.append(currObjectiveValue)
    return mu, pointsToCentroid, lO