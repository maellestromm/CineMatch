import numpy as np
import letterboxd_spider_rich_content

class KNN:

    def euclidDist(self, x1, x2):
        # X1 and x2 are 1D numpy arrays
        # Returns: The euclidean distance between x1 and x2
        dimensionDiffs = x1 - x2
        dimensionDiffs = dimensionDiffs ** 2
        dist = np.sum(dimensionDiffs)
        dist = np.sqrt(dist)
        return dist

    def getMean(self, data):
        # Data is a 2D numpy array where each row contains data for a movie
        # Returns: An array containing the mean of each column of data
        n = len(data)
        means = np.zeros(len(data[0]))
        if n == 0:
            print("No data given")
            return np.array([])
        for row in data:
            means += row
        means /= n
        return means

    def getStdDev(self, data):
        # Data is a 2D numpy array where each row contains data for a movie
        # What each column represents is currently undecided, but shouldn't matter for any of the functions here
        # Returns: An array containing the standard deviation of each column of data
        n = len(data)
        means = self.getMean(data)
        stdDevs = np.zeros(len(data[0]))
        if n == 0:
            print("No data given")
            return []
        for row in data:
            stdDevs += (row - means) ** 2
        stdDevs /= n
        stdDevs = np.sqrt(stdDevs)
        return stdDevs
    
    def standardStandardizer(self, data):
        # Data is a 2D numpy array where each row contains data for a movie
        # Returns: Data, but all values have been standardized
        # The reason it's called standardStandardizer is because we might want different columns to have different weights in the future
        # but this function just standardizes all the values using the regular standardization formula
        means = self.getMean(data)
        stdDevs = self.getStdDev(data)
        newData = np.copy(data)
        for i in range(len(data)):
            newData[i] = (newData[i] - means) / stdDevs
        return newData
    
    def knn(self, data, val, k):
        # Data is a 2D numpy array where each row contains data for a movie
        # Val is a 1D numpy array containing data for... something. A movie we want to find similar movies for? The average movie data of a user's watch history?
        # K is k
        # Either way, this function returns indexes (indices?) of the k movies in data closest to val
        dists = np.array([])
        for row in data:
            dist = self.euclidDist(row, val)
            dists = np.append(dists, dist) # I perhaps may not have thought these variable names through. whoops
        knns = np.array([])
        for i in range(k):
            if i >= len(dists):
                # if k is larger than the rows in data, just return everything. it's in order of closest to largest now which is cool ig
                return knns
            smallestIndex = 0
            for j in range(len(dists)): #looks for smallest distance that's not already in knns
                if dists[j] < dists[smallestIndex] and j not in knns:
                    smallestIndex = j
            knns = np.append(knns, smallestIndex)
        return knns